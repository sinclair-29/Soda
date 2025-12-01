from transformer_lens import HookedTransformer
import torch

def load_target_model(model_name, device):
    print(f"Loading model: {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model = model.eval()
    return model