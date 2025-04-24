import os
import json
from typing import List, Tuple

# default output length for LLM
default_max_output_length = 256
default_min_prompt_length = 4

# get current abs path
current_file_path = os.path.dirname(__file__)
# print(f"当前文件的绝对路径是: {current_file_path}")
request_root_dir = current_file_path + "/../../requests/"

def read_chatGPT(file_path):
    # Load the dataset.
    requests = []
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        for data in dataset:
            context = data["conversations"][0]["value"]
            answer = data["conversations"][1]["value"]
            requests.append((context, answer))
    return requests

def read_txt(file_path, max_num=-1):
    # 从txt文件中读取请求；每行一个request
    requests = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            req = line.strip()
            if req:
                requests.append((req, ""))
                if max_num != -1 and len(requests) > max_num:
                    break
    return requests
    
def load_requests(
        req_file, 
        tokenizer, 
        max_embedding_positions, 
        max_nums=-1
    ) -> List[Tuple[str, int, int]]:
    requests = []
    file_path = request_root_dir + req_file
    if not os.path.exists(file_path):
        print(f"文件不存在, 请检查路径：{file_path}")
        return requests

    # 处理逻辑，根据数据集确定
    if "GPT" in req_file:
        requests = read_chatGPT(file_path)
    else:
        requests = read_txt(file_path, max_nums)

    # tokenizers 处理
    prompts = [prompt for prompt, _ in requests]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in requests]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(requests)):
        output_len = len(completion_token_ids[i])
        # 当数据集未提供output时，或者output过于短时
        if output_len < 4:
            output_len = default_max_output_length
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long snd too short equences. (select 512 ~ 2k)
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < default_min_prompt_length:
            continue
        if prompt_len > max_embedding_positions:
            continue
        # print(prompt_len)
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset

if __name__ == "__main__":
    from load_parameters import load_model

    _, tokenizor = load_model("Qwen/Qwen1.5-MoE-A2.7B")

    requests = load_requests("requestM2.txt", tokenizor, max_embedding_positions=256)

    print(len(requests))
    print(requests[0])
