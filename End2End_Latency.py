import os
import time
import torch
import warnings
from utils.load_parameters import load_model
from utils.load_requests import load_requests
from BatchSize_SelectedExperts import get_selected_experts
from Distance_SelectedExperts import calculate_metrics

warnings.filterwarnings('ignore')

USE_LRU_POLICY=True     # 是否使用新policy？ false表示使用新policy
USE_CACHE_AWARE=False    # 是否选择 cacher aware expert selection?
USE_NEW_MODEL=False

def main(model_name, req_file, max_embed_tokens, output_path):    
    print("开始加载模型...")
    model, tokenizer = load_model(model_name)
    if model == None or tokenizer == None:
        return 
    print("模型加载完成！")

    # 设置policy
    model.set_manager_policy(use_lru_policy=USE_LRU_POLICY)
    # 设置routing
    model.set_manager_aware(use_cache_aware=USE_CACHE_AWARE)
    # 设置使用新model
    if USE_NEW_MODEL:
        if "Qwen1.5" in model_name:
            model.use_delta_router("Qwen1.5")
        elif "Qwen2" in model_name:
            model.use_delta_router("Qwen2")
        else:
            model.use_delta_router("DeepSeek")

    print("开始读取requests")
    requests = load_requests(req_file, tokenizer, max_embed_tokens)
    # [(req, output), ] -> [req, ]
    requests = [req[0].strip() for req in requests if req[0].strip()]
    print("request读取完成！")

    overall_end2end_file = output_path + "/overall_end2end_lat.txt"
    f_wait_lat = open(overall_end2end_file, "w", encoding="utf-8")
    
    # warm up
    for idx in range(5):
        req = requests[idx]
        inputs = tokenizer(req, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_router_logits=True,   # 启用路由器 logits 输出
                predict_in_future_s=2,       # defalut 跨1层
                return_dict=True,
                use_cache=False,             # 防止累积状态
            )

    # inference
    start_time = time.perf_counter()
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
                predict_in_future_s=1,       # <--- 指定跨层预测偏移
                return_dict=True,
                use_cache=False            # 防止累积状态
            )

        
    # calc and store avg waiting latency
    end2end_lat = time.perf_counter() - start_time
    f_wait_lat.write(f"\nOverall-Latency: {end2end_lat:.4f}\n")

    f_wait_lat.close()
        

if __name__ == "__main__":
    
    # Models = ["Qwen/Qwen1.5-MoE-A2.7B", \
    #           "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4", \
    #           "deepseek-ai/DeepSeek-V2-Lite", \
    #           "OPEA/DeepSeek-V2.5-1210-int4-sym-inc"]

    # Models = ["Qwen/Qwen1.5-MoE-A2.7B", "deepseek-ai/DeepSeek-V2-Lite"]
    Models = ["deepseek-ai/DeepSeek-V2-Lite"]

    ResFilePath = "./Mot_Test_Res/Base_End2End_Latency"
    if not USE_LRU_POLICY and USE_CACHE_AWARE and USE_NEW_MODEL:
        ResFilePath = "./Mot_Test_Res/Overall_End2End_Latency"
    elif USE_NEW_MODEL:
        ResFilePath = "./Mot_Test_Res/NewModel_End2End_Latency"
    elif not USE_LRU_POLICY:
        ResFilePath = "./Mot_Test_Res/NewPolicy_End2End_Latency"
    elif USE_CACHE_AWARE:
        ResFilePath = "./Mot_Test_Res/CacheAware_End2End_Latency"
    
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
        
        model_tag = model_name.split("/")[-1]
        model_res_file_path = ResFilePath + "/" + model_tag
        if not os.path.exists(model_res_file_path):
            os.makedirs(model_res_file_path)

        main(model_name, "requests_BatchSize_numExperts_ShareGPT.json", 2048, model_res_file_path)
