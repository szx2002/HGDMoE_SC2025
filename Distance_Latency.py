import os
import torch
import warnings
from utils.load_parameters import load_model
from utils.load_requests import load_requests
from BatchSize_SelectedExperts import get_selected_experts
from Distance_SelectedExperts import calculate_metrics

warnings.filterwarnings('ignore')

def main(model_name, req_file, max_embed_tokens, output_root_path):    
    print("开始加载模型...")
    model, tokenizer = load_model(model_name)
    if model == None or tokenizer == None:
        return 
    print("模型加载完成！")

    print("开始读取requests")
    requests = load_requests(req_file, tokenizer, max_embed_tokens)
    # [(req, output), ] -> [req, ]
    requests = [req[0].strip() for req in requests if req[0].strip()]
    print("request读取完成！")

    # warm up
    for idx in range(5):
        req = requests[idx]
        inputs = tokenizer(req, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_router_logits=True,   # 启用路由器 logits 输出
                predict_in_future_s=1,       # defalut 跨1层
                return_dict=True,
                use_cache=False              # 防止累积状态
            )

    # 循环10次
    for iter in range(10):
        model_tag = model_name.split("/")[-1]
        output_path = output_root_path + f"/{model_tag}_{iter}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        overall_avg_waiting_file = output_path + "/overall_waiting_lat-Steps.txt"
        overall_avg_cache_mis_file = output_path + "/overall_cache_mis_lat-Steps.txt"
        f_wait_lat = open(overall_avg_waiting_file, "w", encoding="utf-8")
        f_cache_mis_lat = open(overall_avg_cache_mis_file, "w", encoding="utf-8")

        # 遍历 NumSteps
        for numSteps in range(1, min(2, model.config.num_hidden_layers)):
            # 文件名定义 - record selected experts
            experts_output_file = output_path + "/activated_experts-Steps_" + str(numSteps) + ".txt"
            
            # 初始化累积变量
            all_waiting_latency = []
            all_cache_mis_latency = []
            all_cache_mis_rate = []
        
            # 打开统计文件，包括 activated_experts.txt 和 acc_layer.txt
            with open(experts_output_file, 'w', encoding='utf-8') as exp_file:
                exp_file.write(f"num_tokens\tavg_euclidean_distance\tavg_cosine_similarity\t\
                                sum_euclidean_distance\ttotal_selected_experts\twait_latency\t\
                                cache_mis_latency\tcache_mis_rate\n")
                for req_idx, req in enumerate(requests):
                    # 重置 model latency_buffer
                    model.reset_latency()
                    model.enable_latency_recording()

                    inputs = tokenizer(req, return_tensors="pt").to("cuda")
                    
                    with torch.inference_mode():
                        # 注意 predict_in_future_s > 0
                        outputs = model(
                            **inputs,
                            output_router_logits=True,  # 启用路由器 logits 输出
                            predict_in_future_s=numSteps,       # <--- 指定跨层预测偏移
                            return_dict=True,
                            use_cache=False            # 防止累积状态
                        )

                    # get_distance by using token_embeddings
                    token_embeddings = outputs.last_hidden_state.squeeze(0)
                    avg_euclidean_distance, avg_cosine_similarity, sum_euclidean_distance = calculate_metrics(token_embeddings)
                    
                    # get waiting latency for MOE swap_in
                    waiting_latency = outputs.get("wait_latency", None)
                    cache_mis_latency = outputs.get("cache_mis_latency", None)
                    cache_mis_rate_list = outputs.get("cache_mis_rate", None)
                    if cache_mis_rate_list:
                        cache_mis_rate = sum(cache_mis_rate_list)/len(cache_mis_rate_list)
                        all_cache_mis_rate.append(cache_mis_rate)
                    else:
                        cache_mis_rate = -1
                    
                    # 如果模型输出了waiting latency信息
                    if waiting_latency is not None:
                        print(f"请求 {req_idx+1} => moe swap_in waiting latency: {waiting_latency:.4f}")
                        print(f"请求 {req_idx+1} => moe cache_mis latency: {cache_mis_latency:.4f}")
                        print(f"请求 {req_idx+1} => moe cache mis rate: {cache_mis_rate}")
                        
                        # 累积lat
                        all_waiting_latency.append(waiting_latency)
                        all_cache_mis_latency.append(cache_mis_latency)
                    
                    # 获取激活的 experts
                    # outputs.router_logits 是一个包含每层 router_logits 的元组
                    router_logits_tuple = outputs.router_logits
                    if len(router_logits_tuple) == 0:
                        exp_file.write("0\n")
                        continue
                    # get experts by using outputs.router_logits
                    num_tokens, num_layers, selected_experts = get_selected_experts(model, model_name, outputs)
                
                    num_total = 0
                    for layer_idx in range(num_layers):
                        experts_in_layer = selected_experts[:, layer_idx].reshape(-1).tolist()
                        unique_experts_in_layer = set(experts_in_layer)
                        num_unique_experts = len(unique_experts_in_layer)
                        num_total += num_unique_experts
                    
                    exp_file.write(f"{num_tokens}\t{avg_euclidean_distance}\t{avg_cosine_similarity}\t{sum_euclidean_distance}\t\
                                {num_total}\t{waiting_latency}\t{cache_mis_latency}\t{cache_mis_rate}\n")
            
            # calc and store avg waiting latency
            f_wait_lat.write(f"\nnumSteps={numSteps}\n")
            if all_waiting_latency:
                avg_waiting_latency = sum(all_waiting_latency) / len(all_waiting_latency)
                f_wait_lat.write(f"moe experts swap_in waiting_latency: {avg_waiting_latency:.4f}\n")

            # calc and store avg cache mis latency
            f_cache_mis_lat.write(f"\nnumSteps={numSteps}\n")
            if all_cache_mis_latency:
                avg_cache_mis_latency = sum(all_cache_mis_latency) / len(all_cache_mis_latency)
                f_cache_mis_lat.write(f"moe experts swap_in cache_mis_latency: {avg_cache_mis_latency:.4f}\n")
                avg_cache_mis_rate = sum(all_cache_mis_latency) / len(all_cache_mis_rate)
                f_cache_mis_lat.write(f"moe experts swap_in cache_mis_rate: {avg_cache_mis_rate:.4f}\n")

        f_wait_lat.close()
        f_cache_mis_lat.close()
        

if __name__ == "__main__":
    
    # Models = ["Qwen/Qwen1.5-MoE-A2.7B", \
    #           "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4", \
    #           "deepseek-ai/DeepSeek-V2-Lite", \
    #           "OPEA/DeepSeek-V2.5-1210-int4-sym-inc"]

    Models = ["Qwen/Qwen1.5-MoE-A2.7B", "deepseek-ai/DeepSeek-V2-Lite", "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4"]

    ResFilePath = "./Mot_Test_Res/TokenDis-Latency"
    if not os.path.exists(ResFilePath):
        os.makedirs(ResFilePath)

    for model_name in Models:
        torch.cuda.empty_cache()
    
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"当前显存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
        


        main(model_name, "requests_TokenDis_latency.txt", 2048, ResFilePath)
