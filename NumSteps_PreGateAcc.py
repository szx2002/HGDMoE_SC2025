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
        accuracy_output_file = output_path + "/layer_acc-Step_" + str(numSteps) + ".txt"
        overall_accuracy_file = output_path + "/overall_acc-Steps_" + str(numSteps) + ".txt"
        
        # 初始化累积变量
        all_avg_accuracies = []
        all_layer_accuracies = None  # 稍后根据层数初始化
    
        # 打开统计文件，包括 activated_experts.txt 和 acc_layer.txt
        with open(experts_output_file, 'w', encoding='utf-8') as exp_file, \
                open(accuracy_output_file, 'w', encoding='utf-8') as acc_file:
            
            for req_idx, req in enumerate(requests):
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
                
                # 获取模型输出中的 avg_accuracy 和 all_layer_accuracy
                avg_accuracy = outputs.get("avg_accuracy", None)
                layer_accuracy = outputs.get("all_layer_accuracy", None)
                
                # 如果模型输出了准确率信息
                if avg_accuracy is not None and layer_accuracy is not None:
                    print(f"请求 {req_idx+1} => 平均准确率: {avg_accuracy:.4f}")
                    # 写入 acc_layer.txt
                    acc_file.write(f"Request {req_idx+1} => 平均准确率: {avg_accuracy:.4f}\n")
                    for i, acc_val in enumerate(layer_accuracy):
                        acc_file.write(f"  Layer {i}: {acc_val:.4f}\n")
                    acc_file.write("\n")
                    
                    # 累积准确率
                    all_avg_accuracies.append(avg_accuracy)
                    
                    if all_layer_accuracies is None:
                        # 根据第一条请求的层数初始化
                        num_layers = len(layer_accuracy)
                        all_layer_accuracies = [[] for _ in range(num_layers)]
                    
                    for i, acc_val in enumerate(layer_accuracy):
                        all_layer_accuracies[i].append(acc_val)
                
                # 获取激活的 experts
                # outputs.router_logits 是一个包含每层 router_logits 的元组
                router_logits_tuple = outputs.router_logits
                print("router_logits 是一个 tuple，长度为:", len(router_logits_tuple))
                if len(router_logits_tuple) == 0:
                    print("没有路由器 logits 输出，检查模型配置和输出选项。")
                    exp_file.write("0\n")
                    continue
                # get experts by using outputs.router_logits
                _, num_layers, selected_experts = get_selected_experts(model, model_name, outputs)
                
                # 提取实际激活的专家（top-2）
                actual_experts = []
                for layer_idx in range(num_layers):
                    # 获取该层所有 token 选择的专家
                    experts_in_layer = selected_experts[:, layer_idx].tolist()
                    # 使用 set 去重
                    unique_experts_in_layer = set(experts_in_layer)
                    actual_experts.append(list(unique_experts_in_layer))  # re-write to list
                
                # get_predicted experts; which is been stored at model.predictions_buffer
                # predictions_buffer[j] 存储的是预测 layer j 的未来层 i = j + numSteps 的专家
                predicted_experts_list = []
                for layer_idx in range(len(router_logits_tuple)):
                    j = layer_idx - numSteps
                    if j >= 0 and getattr(model, 'predictions_buffer', None) and model.predictions_buffer[j] is not None:
                        # predictions_buffer[j] 是 [batch*seq_len, 2]
                        predicted = model.predictions_buffer[j].cpu().numpy().tolist()
                        predicted_experts_list.append(predicted)
                    else:
                        # 如果 j < 0 或者没有预测，填充空列表
                        predicted_experts_list.append([])
                
                # 写入 activated_experts.txt
                # 计算每层激活的专家数量（唯一专家数量）
                print(f"\n请求 {req_idx + 1}: {req}")
                print("每一层中激活的专家数量以及请求中激活的专家总数:")
                
                total_activated_experts = set()
                num_total = 0

                for layer_idx in range(num_layers):
                    experts_in_layer = selected_experts[:, layer_idx].reshape(-1).tolist()
                    unique_experts_in_layer = set(experts_in_layer)
                    num_unique_experts = len(unique_experts_in_layer)
                    num_total += num_unique_experts
                    print(f"  Layer {layer_idx}: 激活的专家数量 (top-k) = {num_unique_experts}")
                    total_activated_experts.update(unique_experts_in_layer)
                
                print(f"  该请求中激活的专家总数 = {num_total}\n")
                # 写入 activated_experts.txt
                exp_file.write(f"{num_total}\n")
        
        # 计算并保存总体平均准确率
        if all_avg_accuracies and all_layer_accuracies:
            overall_avg_accuracy = sum(all_avg_accuracies) / len(all_avg_accuracies)
            overall_layer_accuracies = [
                sum(layer_acc) / len(layer_acc) if layer_acc else 0.0 
                for layer_acc in all_layer_accuracies
            ]
            
            print(f"\n所有请求的平均准确率: {overall_avg_accuracy:.4f}")
            # 打开 overall_acc.txt 进行写入
            with open(overall_accuracy_file, 'w', encoding='utf-8') as overall_acc_file:
                overall_acc_file.write(f"所有请求的平均准确率: {overall_avg_accuracy:.4f}\n")
                for i, acc_val in enumerate(overall_layer_accuracies):
                    overall_acc_file.write(f"  Layer {i} 的平均准确率: {acc_val:.4f}\n")
            
            print("总体平均准确率已保存到 overall_acc.txt 文件中。")
        else:
            print("没有足够的数据来计算总体平均准确率。")
        

if __name__ == "__main__":
    
    # Models = ["Qwen/Qwen1.5-MoE-A2.7B", \
    #           "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4", \
    #           "deepseek-ai/DeepSeek-V2-Lite", \
    #           "OPEA/DeepSeek-V2.5-1210-int4-sym-inc"]

    Models = ["deepseek-ai/DeepSeek-V2-Lite"]

    ResFilePath = "./Mot_Test_Res/NumSteps-PreGateAcc"
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

        main(model_name, "requestM3.txt", 2048, model_res_file_path)
