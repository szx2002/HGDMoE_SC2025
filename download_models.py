# download models from huggingface
import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = {
    "Qwen/Qwen1.5-MoE-A2.7B": "../huggingfaceQwen1.5-Moe-14B-A2.7B",
    "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4": "../huggingfaceQwen2-Moe-57B-A14B-Int4",
    "deepseek-ai/DeepSeek-V2-Lite": "../huggingfaceDeepseekV2-Lite",
    "OPEA/DeepSeek-V2.5-1210-int4-sym-inc": "../huggingfaceDeepseekV2.5",
    # "OPEA/DeepSeek-V3-int4-sym-gptq-inc": "../huggingfaceDeepseekV3"
}

# download deepseek-vl2
for model_name, local_cache_dir in MODEL_NAME.items():
    print(f"start download model: {model_name} to local path: {local_cache_dir}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_cache_dir, trust_remote_code=True, torch_dtype=torch.float16)

    print(f"model {model_name} is stored at: {local_cache_dir}")

