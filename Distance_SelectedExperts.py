import os
import torch
import warnings
import numpy as np
from utils.load_parameters import load_model
from utils.load_requests import load_requests
from BatchSize_SelectedExperts import get_selected_experts

warnings.filterwarnings('ignore')

# 计算相邻 token 向量之间的欧几里得距离、余弦相似度及欧几里得距离总和
def calculate_metrics(embeddings):
    if len(embeddings) < 2:
        return 0, 0, 0  # 如果 token 少于 2 个，返回 0
    
    euclidean_distances = [torch.dist(embeddings[i], embeddings[i + 1]).item() \
                           for i in range(len(embeddings) - 1)]
    cosine_similarities = [
        torch.nn.functional.cosine_similarity(embeddings[i], embeddings[i + 1], dim=0).item() 
        for i in range(len(embeddings) - 1)
    ]
    
    avg_euclidean_distance = np.mean(euclidean_distances)
    avg_cosine_similarity = np.mean(cosine_similarities)
    sum_euclidean_distance = np.sum(euclidean_distances)
    
    return avg_euclidean_distance, avg_cosine_similarity, sum_euclidean_distance


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

    # 打开 activated_experts.txt 文件以写入模式
    with open(output_path, 'w', encoding='utf-8') as output_file:
        # 四列
        output_file.write(f"num_tokens\tavg_euclidean_distance\tavg_cosine_similarity\t\
                          sum_euclidean_distance\ttotal_selected_experts\n")
        for req_idx, req in enumerate(requests):
            # 对请求进行分词并获取 token 向量
            inputs = tokenizer(req, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_router_logits=True,  # 启用路由器 logits 输出
                    return_dict=True,
                    use_cache=False  # 防止累积状态
                )
            
            # get_distance by using token_embeddings
            token_embeddings = outputs.last_hidden_state.squeeze(0)
            avg_euclidean_distance, avg_cosine_similarity, sum_euclidean_distance = calculate_metrics(token_embeddings)

            # get the overall selected experts for req
            router_logits_tuple = outputs.router_logits
            # 确保 tuple 非空
            if len(router_logits_tuple) == 0:
                print("没有路由器 logits 输出，检查模型配置和输出选项。")
                # 写入 0 作为该请求的激活专家总数
                output_file.write(f"{avg_euclidean_distance}\t{avg_cosine_similarity}\t{sum_euclidean_distance}\t-1\n")
                continue
            
            # get selected experts
            num_tokens, num_layers, selected_experts = get_selected_experts(model, model_name, outputs)
            
            num_total = 0
            for layer_idx in range(num_layers):
                # 获取该层所有 token 选择的专家
                experts_in_layer = selected_experts[:, layer_idx].tolist()
                # 使用 set 去重
                unique_experts_in_layer = set(experts_in_layer)
                num_unique_experts = len(unique_experts_in_layer)
                num_total += num_unique_experts
            
            # store num_total 
            output_file.write(f"{num_tokens}\t{avg_euclidean_distance}\t{avg_cosine_similarity}\t{sum_euclidean_distance}\t{num_total}\n")
    

if __name__ == "__main__":
    
    # Models = ["Qwen/Qwen1.5-MoE-A2.7B", \
    #           "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4", \
    #           "deepseek-ai/DeepSeek-V2-Lite", \
    #           "OPEA/DeepSeek-V2.5-1210-int4-sym-inc"]

    Models = ["OPEA/DeepSeek-V2.5-1210-int4-sym-inc"]

    ResFilePath = "./Mot_Test_Res/TokenDis-SelectedExperts"
    if not os.path.exists(ResFilePath):
        os.makedirs(ResFilePath)

    for model_name in Models:
        torch.cuda.empty_cache()

        print(model_name)
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"当前显存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
        
        model_tag = model_name.split("/")[-1]
        model_res_file_path = ResFilePath + "/activated_experts-" + model_tag + ".txt"

        main(model_name, "requests_BatchSize_numExperts_ShareGPT.json", 2048, model_res_file_path)
