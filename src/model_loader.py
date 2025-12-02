from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


def load_target_model(model_name, device):
    print(f"Loading model from: {model_name}...")

    # 判定是否为本地路径
    if os.path.isdir(model_name) or "/" in model_name:
        try:
            # 1. 自动加载原生 HF 模型 (不强制 float16，兼容 CPU)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                # 如果是 CPU，默认 float32 更稳；如果是 CUDA，AutoModel 会自动处理
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 2. 关键修改：根据路径名判断架构类型
            # TransformerLens 需要知道原始架构名称才能正确映射权重
            lower_name = model_name.lower()

            if "tiny" in lower_name:
                # 如果路径里有 tiny，告诉库这是 TinyStories
                official_name = "roneneldan/TinyStories-33M"
            elif "llama" in lower_name:
                # 如果路径里有 llama，告诉库这是 Llama-2
                official_name = "meta-llama/Llama-2-7b-hf"
            else:
                # 默认尝试 GPT2
                official_name = "gpt2"

            print(f"Mapping local model to architecture: {official_name}")

            # 3. 封装进 HookedTransformer
            model = HookedTransformer.from_pretrained(
                official_name,  # 这里传入架构名，而不是本地路径
                hf_model=hf_model,
                tokenizer=tokenizer,
                device=device,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False
            )
        except Exception as e:
            print(f"Local loading failed: {e}")
            raise e
    else:
        # 非本地路径，直接走默认逻辑
        model = HookedTransformer.from_pretrained(model_name, device=device)

    model = model.eval()
    return model