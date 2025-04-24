import json
import random
from transformers import AutoTokenizer
from utils.load_requests import read_chatGPT

max_length = 2048
num_requests = 50

def write_chatGPT(file_path, request_list):
    """
    Write a new entry with the given context and answer to a JSON file.
    The format of the JSON file matches the format used in `read_chatGPT`.
    """
    try:
        # Read the existing data if the file exists
        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty dataset
        dataset = []

    for request in request_list:
        context, answer = request
        new_entry = {
            "conversations": [
                {"value": context},
                {"value": answer}
            ]
        }

        # Append the new entry
        dataset.append(new_entry)

    # Write the updated dataset back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def parse_share_gpt(gpt_file, outfile, tokenizer, need_diff_length = False):
    # read share_gpt
    requests = read_chatGPT(gpt_file)
    # 打乱顺序
    random.shuffle(requests)
    
    # tokenizers 处理
    prompts = []
    answers = []
    for i in range(len(requests)):
        prompt, answer = requests[i]
        if len(prompt.split(" ")) > max_length:
            continue
        prompts.append(prompt)
        answers.append(answer)
    prompt_token_ids = tokenizer(prompts).input_ids

    print(len(prompts))

    # get selected prompts
    selected_requests = []
    # if need diff length
    selected_diff_length = {i : [] for i in [8, 16, 32, 64, 128, 256, 512, 1024]}
    for i in range(len(prompts)):
        # 获取不同request长度的requests，测试batch_size + token-distance
        if need_diff_length:
            prompt = prompts[i]
            answer = answers[i]
            prompt_len = len(prompt_token_ids[i])
            if prompt_len > max_length:
                continue
            if prompt_len >= 4 and prompt_len < 12:
                if len(selected_diff_length[8]) > num_requests:
                    continue
                selected_diff_length[8].append((prompt, answer))
            elif prompt_len >= 12 and prompt_len < 24:
                if len(selected_diff_length[16]) > num_requests:
                    continue
                selected_diff_length[16].append((prompt, answer))
            elif prompt_len >= 24 and prompt_len < 48:
                if len(selected_diff_length[32]) > num_requests:
                    continue
                selected_diff_length[32].append((prompt, answer))
            elif prompt_len >= 48 and prompt_len < 96:
                if len(selected_diff_length[64]) > num_requests:
                    continue
                selected_diff_length[64].append((prompt, answer))
            elif prompt_len >= 96 and prompt_len < 192:
                if len(selected_diff_length[128]) > num_requests:
                    continue
                selected_diff_length[128].append((prompt, answer))
            elif prompt_len >= 192 and prompt_len < 384:
                if len(selected_diff_length[256]) > num_requests:
                    continue
                selected_diff_length[256].append((prompt, answer))
            elif prompt_len >= 384 and prompt_len < 768:
                if len(selected_diff_length[512]) > num_requests:
                    continue
                selected_diff_length[512].append((prompt, answer))
            elif prompt_len >= 768 and prompt_len < 1536:
                if len(selected_diff_length[1024]) > num_requests:
                    continue
                selected_diff_length[1024].append((prompt, answer))
        else:
            prompt = prompts[i]
            answer = answers[i]
            prompt_len = len(prompt_token_ids[i])
            if prompt_len > max_length:
                continue
            if prompt_len < 0.5 * max_length:
                continue
            print(prompt_len)
            selected_requests.append((prompt, answer))
            if len(selected_requests) > num_requests:
                break
    
    if need_diff_length:
        for idx in selected_diff_length:
            selected_requests.extend(selected_diff_length[idx])

    write_chatGPT(outfile, selected_requests)

if __name__ == "__main__":
    # tokenizer - limit prompt length
    tokenizer = AutoTokenizer.from_pretrained(
        "../huggingfaceDeepseekV2-Lite/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0", 
        trust_remote_code=True)
    
    gpt_file = "../requests/ShareGPT_V3_unfiltered_cleaned_split.json"

    # num_steps vs latency
    # outfile = "../requests/requests_NumSteps_Latency_ShareGPT.json"
    # parse_share_gpt(gpt_file, outfile, tokenizer, False)

    # batch_size vs distance + num_experts
    outfile = "../requests/requests_BatchSize_numExperts_shareGPT.json"
    parse_share_gpt(gpt_file, outfile, tokenizer, True)
