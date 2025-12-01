import os
import pickle
import torch
import random
from tqdm import tqdm

def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(path, device="cpu"):
    with open(path, 'rb') as file:
        return pickle.load(file).to(device)

def generate_data(model, cfg, device):
    print("Generating dataset files...")
    
    # 1. True Tokens
    tokens_list = []
    for _ in tqdm(range(cfg.num_targets), desc="Generating Targets"):
        tokens = torch.randint(0, len(model.tokenizer.vocab), (1, cfg.input_len)).to(device)
        tokens_list.append(tokens)
    true_tokens = torch.cat(tokens_list, dim=0).to(device)
    save_pickle(true_tokens, f'data/true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl')

    # 2. Greedy Outputs (Only needed for text inversion)
    # Check if output_len is in cfg to decide if we generate outputs
    if hasattr(cfg, 'output_len') and cfg.output_len:
        print("Generating greedy outputs...")
        batch_size = 100
        all_output_tokens = []
        for batch in range(0, cfg.num_targets, batch_size):
            input_tokens = true_tokens[batch:batch+batch_size].to(device)
            # Note: stop_at_eos=False is important from notebook
            output_tokens = model.generate(
                input_tokens,
                max_new_tokens=cfg.output_len,
                do_sample=False,
                stop_at_eos=False,
                verbose=False,
                return_type="tokens",
            )[:, cfg.input_len:]
            all_output_tokens.append(output_tokens)
        
        all_output_tokens = torch.cat(all_output_tokens, dim=0)
        save_pickle(all_output_tokens, f"data/true_tokens_{cfg.num_targets}_{cfg.input_len}_{cfg.output_len}_greedy.pkl")

    # 3. Initial Tokens (Random/Zeros for start)
    tokens_list = []
    for _ in range(cfg.num_targets):
        tokens = torch.randint(0, len(model.tokenizer.vocab), (1, cfg.input_len)).to(device)
        tokens_list.append(tokens)
    init_tokens = torch.cat(tokens_list, dim=0).to(device)
    save_pickle(init_tokens, f"data/initial_tokens_{cfg.num_targets}_{cfg.input_len}.pkl")