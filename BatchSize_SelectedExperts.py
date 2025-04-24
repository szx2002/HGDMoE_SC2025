import os
import torch
import warnings
import torch.nn.functional as F
from utils.load_parameters import load_model
from utils.load_requests import load_requests

warnings.filterwarnings('ignore')

def DeepSeek_MoeGate(model, router_logits_tuple, last_hidden_state):
    selected_experts = []

    bsz, seq_len, h = last_hidden_state.shape
    for logits in router_logits_tuple:
        # use softmax
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # select topk
        if model.config.topk_method == "greedy":
            _, topk_idx = torch.topk(
                scores, k=model.config.num_experts_per_tok, dim=-1, sorted=False
            )
        elif model.config.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, model.config.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k = model.config.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, model.config.n_group, model.config.n_routed_experts // model.config.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(
                tmp_scores, k=model.config.num_experts_per_tok, dim=-1, sorted=False
            )

        selected_experts.append(topk_idx)
    
    return selected_experts

def Qwen_MoeGate(model, router_logits_tuple):
    selected_experts = []

    for logits in router_logits_tuple:
        # use softmax
        routing_weights = F.softmax(logits, dim=1, dtype=torch.float) 
        _, topk_idx = torch.topk(routing_weights, model.config.num_experts_per_tok, dim=-1) 

        selected_experts.append(topk_idx)
    
    return selected_experts

def get_selected_experts(model, model_name, outputs):
    router_logits_tuple = outputs.router_logits
    # 检查每个元素的形状 [batch_size, seq_length, hidden_states]
    # example_shape = router_logits_tuple[0].shape
    # print(f"每个 router_logits 元素的形状: {example_shape}")

    # 这里不同模型使用方法不一样
    if "DeepSeek" in model_name:
        # use softmax
        selected_experts = DeepSeek_MoeGate(model, router_logits_tuple, outputs.last_hidden_state)
    
        # stack 成 [sequence_length, top_k, num_layers]
        selected_experts = torch.stack(selected_experts, dim=-1) 
        # print("selected_experts shape:", selected_experts.shape)
        
        num_tokens, _, num_layers = selected_experts.shape
        # reshape [seq_len * top_k, num_layer]
        selected_experts = selected_experts.reshape(-1, num_layers)
    elif "Qwen" in model_name:
        # use softmax
        selected_experts = Qwen_MoeGate(model, router_logits_tuple)
    
        # stack 成 [sequence_length, top_k, num_layers]
        selected_experts = torch.stack(selected_experts, dim=-1) 
        # print("selected_experts shape:", selected_experts.shape)
        
        num_tokens, _, num_layers = selected_experts.shape
        # reshape [seq_len * top_k, num_layer]
        selected_experts = selected_experts.reshape(-1, num_layers)
    else:
        # default: Mixtral
        selected_experts = [logits.argmax(dim=-1) for logits in router_logits_tuple]
        # stack 成 [sequence_length, num_layers]
        selected_experts = torch.stack(selected_experts, dim=-1)
        # print("selected_experts shape:", selected_experts.shape)
        num_tokens, num_layers = selected_experts.shape
    
    return num_tokens, num_layers, selected_experts

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

    # per-layer
    per_layer_output = output_path.split(".txt")[0] + "-per_layer.txt"
    # 打开 activated_experts.txt 文件以写入模式
    with open(output_path, 'w', encoding='utf-8') as output_file, \
        open(per_layer_output, 'w', encoding="utf-8") as output_layer:
        output_file.write(f"num_tokens\ttotal_selected_experts\n")
        for req_idx, req in enumerate(requests):
            inputs = tokenizer(req, return_tensors="pt").to("cuda")
            
            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_router_logits=True,  # 启用路由器 logits 输出
                    return_dict=True,
                    use_cache=False  # 防止累积状态
                )
            
            # outputs.router_logits 是一个包含每层 router_logits 的元组
            router_logits_tuple = outputs.router_logits
            print("router_logits 是一个 tuple，长度为:", len(router_logits_tuple))
            
            # 确保 tuple 非空
            if len(router_logits_tuple) == 0:
                print("没有路由器 logits 输出，检查模型配置和输出选项。")
                # 写入 0 作为该请求的激活专家总数
                output_file.write("-1\t0\n")
                continue
            
            # get selected experts
            num_tokens, num_layers, selected_experts = get_selected_experts(model, model_name, outputs)

            print(f"\n请求 {req_idx + 1}; num_tokens: {num_tokens}")
            
            num_total = 0
            output_layer.write(f"\nNum of tokens/batch_size is: {num_tokens}\n")
            for layer_idx in range(num_layers):
                # 获取该层所有 token 选择的专家
                experts_in_layer = selected_experts[:, layer_idx].tolist()
                # 使用 set 去重
                unique_experts_in_layer = set(experts_in_layer)
                num_unique_experts = len(unique_experts_in_layer)
                num_total += num_unique_experts
                output_layer.write(f"Layer {layer_idx}: 激活的专家数量 = {num_unique_experts}\n")
            
            # 将 num_total 写入 activated_experts.txt 文件，确保每行只有一个数字
            output_file.write(f"{num_tokens}\t{num_total}\n")
    

if __name__ == "__main__":
    
    # Models = ["Qwen/Qwen1.5-MoE-A2.7B", \
    #           "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4", \
    #           "deepseek-ai/DeepSeek-V2-Lite", \
    #           "OPEA/DeepSeek-V2.5-1210-int4-sym-inc"]

    Models = ["Qwen/Qwen1.5-MoE-A2.7B", \
              "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4", \
              "deepseek-ai/DeepSeek-V2-Lite"]

    ResFilePath = "./Mot_Test_Res/BatchSize-SelectedExperts"
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
        model_res_file_path = ResFilePath + "/activated_experts-" + model_tag + ".txt"

        main(model_name, "random_text_data.txt", 2048, model_res_file_path)
