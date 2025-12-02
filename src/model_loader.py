from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_target_model(model_name, device):
    print(f"Loading model from: {model_name}...")

    if "/" in model_name:
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Llama-2 建议使用 fp16 节省显存
                device_map=device  # 自动映射到设备
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = HookedTransformer.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                hf_model=hf_model,
                tokenizer=tokenizer,
                device=device,
                dtype=torch.float16
            )
        except Exception as e:
            print(f"本地加载失败，尝试直接加载: {e}")
            model = HookedTransformer.from_pretrained(model_name, device=device)
    else:
        model = HookedTransformer.from_pretrained(model_name, device=device)
    model = model.eval()
    return model