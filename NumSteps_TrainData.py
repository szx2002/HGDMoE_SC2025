import os
import torch
import warnings
from utils.load_parameters import load_model
from utils.load_requests import load_requests
from BatchSize_SelectedExperts import get_selected_experts

warnings.filterwarnings('ignore')

def main(model_name, req_file, max_embed_tokens, output_path):    
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

    # 遍历 NumSteps
    for numSteps in range(1, model.config.num_hidden_layers):

        # 文件名定义
        experts_output_file = output_path + "/activated_experts-Steps_" + str(numSteps) + ".txt"
    
        # 打开统计文件，包括 activated_experts.txt 和 acc_layer.txt
        with open(experts_output_file, 'w', encoding='utf-8') as exp_file:
            exp_file.write("token_ids\tlayer_idx\tpredicted_experts\tactual_experts\n")
            for req_idx, req in enumerate(requests):
                inputs = tokenizer(req, return_tensors="pt").to("cuda")
                token_ids = inputs["input_ids"].to("cpu").tolist()
                
                with torch.inference_mode():
                    # 注意 predict_in_future_s > 0
                    model.reset_experts_record()
                    outputs = model(
                        **inputs,
                        output_router_logits=True,  # 启用路由器 logits 输出
                        predict_in_future_s=numSteps,       # <--- 指定跨层预测偏移
                        return_dict=True,
                        use_cache=False            # 防止累积状态
                    )
                
                # 获取激活的 experts
                # outputs.router_logits 是一个包含每层 router_logits 的元组
                router_logits_tuple = outputs.router_logits
                if len(router_logits_tuple) == 0:
                    print("没有路由器 logits 输出，检查模型配置和输出选项。")
                    exp_file.write("0\n")
                    continue

                # predicted experts and actural experts
                for layer_idx in range(len(router_logits_tuple)):
                    predicted_experts = outputs.predicted_experts[layer_idx]
                    actual_experts = outputs.actual_experts[layer_idx]
                    exp_file.write(f"{token_ids}\t{layer_idx}\t{predicted_experts}\t{actual_experts}\n")
        

if __name__ == "__main__":
    
    # Models = ["Qwen/Qwen1.5-MoE-A2.7B", \
    #           "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4", \
    #           "deepseek-ai/DeepSeek-V2-Lite", \
    #           "OPEA/DeepSeek-V2.5-1210-int4-sym-inc"]

    Models = ["Qwen/Qwen1.5-MoE-A2.7B", "deepseek-ai/DeepSeek-V2-Lite", "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4"]

    ResFilePath = "./Mot_Test_Res/NumSteps-TrainData"
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
