import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, \
                            BitsAndBytesConfig, GenerationConfig


# get current abs path
current_file_path = os.path.dirname(__file__)
# print(f"当前文件的绝对路径是: {current_file_path}")
model_root_dir = current_file_path + "/../../"

MODEL_NAME={
    # "huggingfaceM87Bv01": "huggingfaceM87Bv01", 
    "Qwen/Qwen1.5-MoE-A2.7B": "huggingfaceQwen1.5-Moe-14B-A2.7B",                    # num_experts=60 * 24
    "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4": "huggingfaceQwen2-Moe-57B-A14B-Int4",  # num_experts=64 * 28
    "deepseek-ai/DeepSeek-V2-Lite": "huggingfaceDeepseekV2-Lite",                    # num_experts=64 * 26
    "OPEA/DeepSeek-V2.5-1210-int4-sym-inc": "huggingfaceDeepseekV2.5",               # num_experts=160
    # "OPEA/DeepSeek-V3-int4-sym-gptq-inc": "huggingfaceDeepseekV3"                  # num_experts=256 模型超过350GB
}

def load_model(model_name):
    if model_name not in MODEL_NAME:
        print(f"invalid model_name: {model_name}")
        return None, None

    # model 路径
    local_model_dir = model_root_dir + MODEL_NAME[model_name]
    for dir_name in os.listdir(local_model_dir):
        if "models" in dir_name:
            local_model_dir = local_model_dir + "/" + dir_name + "/snapshots/"
    local_model_dir = local_model_dir + os.listdir(local_model_dir)[0]

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir, 
        trust_remote_code=True)
    
    # 设备memory
    max_memory={1: "40GB"}
    # with init_empty_weights():
    #     config = AutoConfig.from_pretrained(local_model_dir)
    #     model = AutoModelForCausalLM.from_config(config)
    # device_map = infer_auto_device_map(model, max_memory={1: "40GB"})
    # print(device_map)
    
    if "OPEA" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            # device_map=device_map,
        )
    elif "Mixtral" in model_name:
        # 使用4-bit量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="fp16"
        )
        
        # 设置设备映射
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            # device_map=device_map,
            torch_dtype=torch.float16,
            max_memory=max_memory,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            # device_map=device_map,
            max_memory=max_memory,
        )
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, tokenizer


if __name__ == "__main__":
    _, _ = load_model("deepseek-ai/DeepSeek-V2-Lite")