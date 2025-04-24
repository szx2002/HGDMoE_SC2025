# coding=utf-8
# support expert swap_in and swap_out
from typing import Optional
import torch
import random
import time
import queue
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

# set different priority for tasks, (lower number => higher priority)
HIGH_PRIORITY=0
LOW_PRIORITY=1

class QuantExpert():
    def __init__(self, gate_proj, up_proj, down_proj):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
    
    def to(self, device, non_blocking: Optional[bool] = False):
        self.gate_proj.to(device, non_blocking)
        self.up_proj.to(device, non_blocking)
        self.down_proj.to(device, non_blocking)

class PriorityTask:
    def __init__(self, 
                 priority, 
                 layer_idx, 
                 expert_id, 
                 task_type, 
                 func):
        self.priority = priority
        self.func = func
        self.layer_idx = layer_idx
        self.expert_id = expert_id
        self.task_type = task_type
        self.future = None 
        self.future_ready = threading.Event()

    def __lt__(self, other):
        return self.priority < other.priority
    
    def set_future(self, future):
        self.future = future
        self.future_ready.set()

    def wait_for_future(self):
        self.future_ready.wait()

    def done(self):
        return self.future is not None and self.future.done()
    
    def result(self):
        # if self.future is None, means the task is not start
        self.wait_for_future()
        x = time.perf_counter()
        self.future.result()
        return time.perf_counter() - x

class PriorityThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers, *args, **kwargs):
        super().__init__(max_workers, *args, **kwargs)
        self.task_queue = queue.PriorityQueue()

    def submit(self, task: PriorityTask):
        """
        set different priority for tasks, (lower number => higher priority)
        """
        self.task_queue.put(task)
        self._process_queue()

    def _process_queue(self):
        while not self.task_queue.empty() and len(self._threads) <= self._max_workers:
            task = self.task_queue.get()
            future = super().submit(task.func)
            task.set_future(future)

class ExpertMemoryManager:
    def __init__(self, 
            num_experts_in_gpu: int,
            num_layers: int,
            memory_threshold: Optional[float] = 0.1,
            max_gpu_memory: Optional[float] = "20GB",
        ):
        """
        only keeps part of experts on GPU device
        """
        self.num_layers = num_layers
        self.num_experts_in_gpu = num_experts_in_gpu
        self.memory_threshold = memory_threshold
        self.max_gpu_memory = self._parse_memory_limit(max_gpu_memory)
        self.gpu_device = None  # be changed in self._init_experts_map
        self.cpu_device = torch.device("cpu")

        # init experts_module_list and cur_experts_device_map
        #   cur_experts_device_map  refers the device info for every experts
        self.experts_module_list = {i: {} for i in range(num_layers)}
        self.cur_experts_device_map = {i: {} for i in range(num_layers)}

        # keep the async-task states
        self.swap_in_tasks_starttime = {i: {} for i in range(num_layers)} # Test
        self.swap_in_tasks = {i: {} for i in range(num_layers)} 
        self.swap_out_tasks = {i: {} for i in range(num_layers)} 
        self.task_queue = queue.PriorityQueue()
        
        self.cuda_stream = torch.cuda.Stream(device=self.gpu_device)
        self.executor = PriorityThreadPoolExecutor(max_workers=2)

        # LRU Cache: tracks access order across all layers
        self.experts_access_order = OrderedDict()
        # New Policy - default skip 1
        self.num_skip_steps = 1
        self.experts_access_order_low = OrderedDict()
        self.experts_access_order_high = OrderedDict()

        self.USE_LRU_POLICY = True
        self.USE_CACHE_AWARE = False

        # init expert map
        self.already_init = False
    
    def use_lru_policy(self, use_lru_policy: bool):
        print(f"{self.USE_LRU_POLICY} vs {use_lru_policy}")
        self.USE_LRU_POLICY = use_lru_policy

    def use_cache_aware(self, use_cache_aware: bool):
        print(f"{self.USE_CACHE_AWARE} vs {use_cache_aware}")
        self.USE_CACHE_AWARE = use_cache_aware

    def _parse_memory_limit(self, memory_str):
        size_map = {"GB": 1024 ** 3, "MB": 1024 ** 2}
        unit = memory_str[-2:].upper()
        value = int(memory_str[:-2])
        return value * size_map.get(unit, 1)

    def _update_access_order(self, layer_idx, expert_id):
        """update LRU access order for an expert"""
        key = (layer_idx, expert_id)
        if self.USE_LRU_POLICY:
            if key in self.experts_access_order:
                self.experts_access_order.move_to_end(key)
            else:
                self.experts_access_order[key] = True
        else:
            # update new policy
            if layer_idx < self.num_skip_steps:
                if key in self.experts_access_order_high:
                    self.experts_access_order_high.move_to_end(key)
                else:
                    self.experts_access_order_high[key] = True
            else:
                if key in self.experts_access_order_low:
                    self.experts_access_order_low.move_to_end(key)
                else:
                    self.experts_access_order_low[key] = True

    def _evict_experts_if_needed(self, num_experts_swap_in):
        "evict experts from GPU based on LRU if exceeding num_experts_in_gpu"
        if self.USE_LRU_POLICY:
            while len(self.experts_access_order) + num_experts_swap_in > self.num_experts_in_gpu:
                # remove the least recently used experts
                # LRU
                (layer_idx, expert_id), _ = self.experts_access_order.popitem(last=False)
                if self.cur_experts_device_map[layer_idx][expert_id] == "cuda":
                    self.offload_to_cpu(layer_idx, expert_id)
        else:
            while len(self.experts_access_order_high) + len(self.experts_access_order_low) \
                + num_experts_swap_in > self.num_experts_in_gpu:
                if self.experts_access_order_low:
                    (layer_idx, expert_id), _ = self.experts_access_order_low.popitem(last=False)
                else:
                    (layer_idx, expert_id), _ = self.experts_access_order_high.popitem(last=False)
                if self.cur_experts_device_map[layer_idx][expert_id] == "cuda":
                    self.offload_to_cpu(layer_idx, expert_id)

    def init_quant_experts_map(self, model):
        model_experts = {i:{} for i in range(28)}
        experts_in_gpu = 0
        # gate_proj
        # up_proj
        # down_proj
        for name, module in model.named_modules():
            if "experts" in name and hasattr(module, "qweight"):
                layer_idx = int(name.split(".")[2])
                expert_id = int(name.split(".")[5])
                proj_name = name.split(".")[6]
                if expert_id not in model_experts[layer_idx]:
                    model_experts[layer_idx][expert_id] = {proj_name: module.qweight}
                else:
                    model_experts[layer_idx][expert_id][proj_name] = module.qweight
                if not self.gpu_device:
                    self.gpu_device = module.qweight.device

        for layer_idx in model_experts:
            experts = model_experts[layer_idx]
            for expert_id in experts:
                expert_info = experts[expert_id]
                if expert_info["gate_proj"] is None:
                    raise(f"Error: no gate_proj in expert {layer_idx}-{expert_id}")
                if expert_info["up_proj"] is None:
                    raise(f"Error: no up_proj in expert {layer_idx}-{expert_id}")
                if expert_info["down_proj"] is None:
                    raise(f"Error: no down_proj in expert {layer_idx}-{expert_id}")
                expert = QuantExpert(expert_info["gate_proj"],
                                    expert_info["up_proj"],
                                    expert_info["down_proj"])
                # keep experts in gpu
                if experts_in_gpu >= self.num_experts_in_gpu:
                    expert.to(torch.device("cpu"))
                    self.cur_experts_device_map[layer_idx][expert_id] = "cpu"
                else:
                    experts_in_gpu += 1
                    self.cur_experts_device_map[layer_idx][expert_id] = "cuda"
                    self._update_access_order(layer_idx, expert_id) # record the experts in GPU
                self.experts_module_list[layer_idx][expert_id] = expert
                # init task states
                self.swap_in_tasks[layer_idx][expert_id] = None
                self.swap_out_tasks[layer_idx][expert_id] = None

    def init_experts_map(self, model):
        """
        initiate
        keep the experts on first {num_experts_in_gpu} experts on GPU, others move to CPU
        """
        try:
            experts_in_gpu = 0
            for name, module in model.named_modules():
                if hasattr(module, "experts"):
                    # get layer_index: name = "layers.12.mlp.1.expert"
                    try:
                        layer_index = int(name.split(".")[1])
                    except ValueError:
                        layer_index = int(name.split(".")[2])
                    experts = module.experts

                    if not self.gpu_device:
                        if not experts or not hasattr(experts[0], 'parameters'):
                            raise ValueError("Experts list is empty or improperly initialized.")
                        try:
                            self.gpu_device = next(experts[0].parameters()).device
                        except StopIteration:
                            self.gpu_device = next(module.gate.parameters()).device

                    for expert_idx, expert in enumerate(experts):
                        # keep experts in gpu
                        if experts_in_gpu >= self.num_experts_in_gpu:
                            expert.to(torch.device("cpu"))
                            self.cur_experts_device_map[layer_index][expert_idx] = "cpu"
                        else:
                            experts_in_gpu += 1
                            self.cur_experts_device_map[layer_index][expert_idx] = "cuda"
                            self._update_access_order(layer_index, expert_idx) # record the experts in GPU
                        self.experts_module_list[layer_index][expert_idx] = expert
                        # init task states
                        self.swap_in_tasks[layer_index][expert_idx] = None
                        self.swap_out_tasks[layer_index][expert_idx] = None
        except:
            self.init_quant_experts(model)
        # finish init
        self.already_init = True

    def check_expert_device(self, layer_idx, expert_idx):
        """
        return the device info for experts
        if swap_in task still running
        """
        return self.cur_experts_device_map[layer_idx][expert_idx]

    def set_expert_device(self, layer_idx, expert_idx, device):
        """set expert device"""
        self.cur_experts_device_map[layer_idx][expert_idx] = device

    def enough_gpu_memory_space(self):
        """check free HBM"""
        torch.cuda.synchronize(self.gpu_device)
        memory_reserved = torch.cuda.memory_reserved(self.gpu_device)
        # memory_free = self.max_gpu_memory - memory_reserved

        return memory_reserved <= self.max_gpu_memory * self.memory_threshold
    
    def _enough_gpu_memory_space(self, num_experts_swap_in):
        """
        wether the gpu memory is enough for experts
        """
        if self.USE_LRU_POLICY:
            return len(self.experts_access_order) + num_experts_swap_in > self.num_experts_in_gpu * 1.2
        else:
            return len(self.experts_access_order_high) + len(self.experts_access_order_low) + \
                num_experts_swap_in > self.num_experts_in_gpu * 1.2

    def _update_device_map(self, layer_idx, expert_id, task_type):
        """update the device_map for expert"""
        if task_type == "swap_in":
            # if device info not match, force transfer
            self.cur_experts_device_map[layer_idx][expert_id] = "cuda"
            self.swap_in_tasks[layer_idx][expert_id] = None
        elif task_type == "swap_out":
            self.cur_experts_device_map[layer_idx][expert_id] = "cpu"
            self.swap_out_tasks[layer_idx][expert_id] = None

    def _update_task_states(self, 
        tar_layer_idx: Optional[int] = -1, 
        tar_expert_id: Optional[int] = -1,
        tar_ops: Optional[str] = "swap_out"
    ) -> None:
        """
        check Async-Offload/Load task finish?
        """
        mat_target_task = False
        new_task_queue = queue.PriorityQueue() # requeue unfinished tasks
        while not self.task_queue.empty():
            task = self.task_queue.get()

            # valid tar_expert, waiting for finish
            if tar_expert_id >= 0 and tar_layer_idx >= 0:
                if task.done():
                    self._update_device_map(task.layer_idx, task.expert_id, task.task_type)
                else:
                    # already get the target expert
                    if mat_target_task:
                        new_task_queue.put(task)
                    else:
                        cache_mis_lat = task.result()
                        self._update_device_map(task.layer_idx, task.expert_id, task.task_type)
                # find the target ops, finish
                if tar_layer_idx == task.layer_idx and tar_expert_id == task.expert_id and task.task_type == tar_ops:
                    mat_target_task = True
            else:
                # just update finished tasks
                if task.done():
                    self._update_device_map(task.layer_idx, task.expert_id, task.task_type)
                else:
                    new_task_queue.put(task)

        self.task_queue = new_task_queue

        # force sync
        torch.cuda.synchronize(self.gpu_device)

        return cache_mis_lat

    def load_to_gpu(self, layer_idx, expert_id, is_high_priority):
        """async load"""
        # update LRU
        self._update_access_order(layer_idx, expert_id)

        def task():
            with torch.cuda.stream(self.cuda_stream):
                self.experts_module_list[layer_idx][expert_id].to(self.gpu_device, non_blocking=True)
                
        self.swap_in_tasks_starttime[layer_idx][expert_id] = time.perf_counter()
        if is_high_priority:
            task_ = PriorityTask(HIGH_PRIORITY, layer_idx, expert_id, "swap_in", task)
        else:
            task_ = PriorityTask(LOW_PRIORITY, layer_idx, expert_id, "swap_in", task)
        self.executor.submit(task_)

        self.swap_in_tasks[layer_idx][expert_id] = task_
        self.task_queue.put(task_)

    def offload_to_cpu(self, layer_idx, expert_id):
        """async-offload"""
        def task():
            with torch.cuda.stream(self.cuda_stream):
                self.experts_module_list[layer_idx][expert_id].to(self.cpu_device, non_blocking=True)

        task_ = PriorityTask(LOW_PRIORITY, layer_idx, expert_id, "swap_out", task)
        self.executor.submit(task_)
        # record
        self.swap_out_tasks[layer_idx][expert_id] = task_
        self.task_queue.put(task_)

    def _iter(self, layer_idx, expert_id_list):
        """
        iterator, return expert_id
        """
        for expert_id in expert_id_list:
            if self.cur_experts_device_map[layer_idx][expert_id] == "cuda":
                yield expert_id

        yield expert_id_list[0]
    

    def wait_for_expert_(self, layer_idx, expert_ids):
        """
        waiting for the asy- expert load finish
        if no task and not in GPU, means cache mis, create an swap_in task
        """
        # update expert_device
        self._update_task_states()

        expert_id = self._iter(layer_idx, expert_ids)

        # update experts access order
        self._update_access_order(layer_idx, expert_id)

        # latency
        wait_lat = 0.0
        cache_mis_lat = 0.0
        start_time = time.perf_counter()

        # if offload task exist and not finished
        if self.swap_out_tasks[layer_idx][expert_id] is not None:
            # print(f"{layer_idx}-{expert_id} offload")
            self._update_task_states(layer_idx, expert_id, "swap_out")

        # check load task
        if self.swap_in_tasks[layer_idx][expert_id] is not None:
            # print(f"{layer_idx}-{expert_id} load")
            self._update_task_states(layer_idx, expert_id, "swap_in")
            wait_lat = time.perf_counter() - start_time
        
        # check device map, if cache mis ~M~U~K~_è®
        if self.check_expert_device(layer_idx, expert_id) != 'cuda':
            # create a swap_in task
            self.prefetch(layer_idx, set([expert_id]), True)
            # wait for task finish
            cache_mis_lat = self._update_task_states(layer_idx, expert_id, "swap_in")
            # cache_mis_lat = time.perf_counter() - start_time

        print(f"{layer_idx}-{expert_id} wait_lat: {wait_lat} \t cache_mis_lat: {cache_mis_lat}")

        return expert_id, wait_lat, cache_mis_lat

    def wait_for_expert(self, layer_idx, expert_id):
        """
        waiting for the asy- expert load finish
        if no task and not in GPU, means cache mis, create an swap_in task
        """
        # update experts access order
        self._update_access_order(layer_idx, expert_id)

        # update expert_device
        self._update_task_states()

        # latency
        wait_lat = 0.0
        cache_mis_lat = 0.0
        start_time = time.perf_counter()

        # if offload task exist and not finished
        if self.swap_out_tasks[layer_idx][expert_id] is not None:
            self._update_task_states(layer_idx, expert_id, "swap_out")

        # check load task
        if self.swap_in_tasks[layer_idx][expert_id] is not None:
            self._update_task_states(layer_idx, expert_id, "swap_in")
            wait_lat = time.perf_counter() - start_time 
            
        # check device map, if cache mis åç¬ç»è®¡
        if self.check_expert_device(layer_idx, expert_id) != 'cuda':
            # create a swap_in task
            self.prefetch(layer_idx, set([expert_id]), True)
            # wait for task finish
            self._update_task_states(layer_idx, expert_id, "swap_in")
            cache_mis_lat = time.perf_counter() - start_time

        print(f"{layer_idx}-{expert_id} wait_lat: {wait_lat} \t cache_mis_lat: {cache_mis_lat}")

        return wait_lat, cache_mis_lat

    def prefetch(self, 
                 layer_idx, 
                 expert_id_list, 
                 is_high_priority:Optional[bool] = False
        ):
        # update expert_device
        self._update_task_states()

        # check free HBM, if exceed threshold, call 
        if self._enough_gpu_memory_space(len(expert_id_list)):
            self._evict_experts_if_needed(len(expert_id_list))

        for expert_id in expert_id_list:
            # whether already in GPU?
            # check whether an offload task exists?
            if self.swap_out_tasks[layer_idx][expert_id] is None and \
                self.cur_experts_device_map[layer_idx][expert_id] == "cuda":
                continue

            # already in tasks?
            if self.swap_in_tasks[layer_idx][expert_id] is not None:
                continue
            # print(f"prefetch: {layer_idx}-{expert_id}: {self.cur_experts_device_map[layer_idx][expert_id]} \t {next(self.experts_module_list[layer_idx][expert_id].parameters()).device}")
            self.load_to_gpu(layer_idx, expert_id, is_high_priority)

    def update_skip_steps(self, num_skip_steps):
        """
        change the skip_steps
        """
        if self.num_skip_steps == num_skip_steps:
            return
        elif self.num_skip_steps > num_skip_steps:
            self.increase_skip_steps(num_skip_steps)
        else:
            self.decrease_skip_steps(num_skip_steps)
        self.num_skip_steps = num_skip_steps 

    def increase_skip_steps(self, num_skip_steps):
        """move access order from low to high"""
        experts_access_order = OrderedDict()
        while self.experts_access_order_low:
            (layer_idx, expert_id), _ = self.experts_access_order_low.popitem(last=False)
            if layer_idx < num_skip_steps:
                self.experts_access_order_high[(layer_idx, expert_id)] = True
            else:
                experts_access_order[(layer_idx, expert_id)] = True
        self.experts_access_order_low = experts_access_order

    def decrease_skip_steps(self, num_skip_steps):
        """move access order from high to low"""
        experts_access_order = OrderedDict()
        while self.experts_access_order_high:
            (layer_idx, expert_id), _ = self.experts_access_order_high.popitem(last=False)
            if layer_idx >= num_skip_steps:
                self.experts_access_order_low[(layer_idx, expert_id)] = True
            else:
                experts_access_order[(layer_idx, expert_id)] = True
        self.experts_access_order_high = experts_access_order


    def sort_selected_experts_qwen(self, layer_idx, num_experts, selected_experts):
        """re-sort the experts list based on whether it is in GPU?"""
        already_in_gpu = []
        swap_in_task_exist = []
        cache_mis = []

        # update expert_device
        self._update_task_states()

        # for expert_id in selected_experts:
        for expert_id in range(num_experts):
            if expert_id not in selected_experts:
                continue
            # already in GPU, && no offload tasks
            if self.cur_experts_device_map[layer_idx][expert_id] == "cuda" and \
                self.swap_out_tasks[layer_idx][expert_id] is None:
                already_in_gpu.append(expert_id)
            elif self.swap_in_tasks[layer_idx][expert_id] is not None:
                swap_in_task_exist.append(expert_id)
            else:
                cache_mis.append(expert_id)
        
        # create prefetch tasks (high priority)
        if cache_mis:
            self.prefetch(layer_idx, cache_mis, True)

        print(f"already_in_gpu: {already_in_gpu}\t cache_mis: {cache_mis}\t swap_in_task_exist: {swap_in_task_exist}")
        if not already_in_gpu:
            # return [i for i in selected_experts]
            return cache_mis + swap_in_task_exist
            # return swap_in_task_exist + cache_mis
        else:
            return already_in_gpu + cache_mis + swap_in_task_exist
    
    def sort_selected_experts_deepseek(self, layer_idx, tokens_per_expert, ep_rank, experts_per_rank):
        start_idx = 0
        already_in_gpu = [] # [(expert_id, start_idx, num_tokens)]
        swap_in_task_exist = []
        cache_mis = []
        cache_mis_experts = []

        # update expert_device
        self._update_task_states()

        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue

            expert_id = i + ep_rank * experts_per_rank

            # already in GPU && no offload tasks
            if self.cur_experts_device_map[layer_idx][expert_id] == "cuda" and \
                self.swap_out_tasks[layer_idx][expert_id] is None:
                already_in_gpu.append((expert_id, start_idx, num_tokens))
            elif self.swap_in_tasks[layer_idx][expert_id] is not None:
                swap_in_task_exist.append((expert_id, start_idx, num_tokens))
            else:
                cache_mis.append((expert_id, start_idx, num_tokens))
                cache_mis_experts.append(expert_id)
            # update start_idx
            start_idx = start_idx + num_tokens

        # create prefetch tasks
        if cache_mis_experts:
            self.prefetch(layer_idx, set(cache_mis_experts), True)

        return already_in_gpu + cache_mis + swap_in_task_exist

if __name__ == "__main__":
    print(1)
