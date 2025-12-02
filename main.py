import argparse
import torch
import json
import os
import pickle
import random
import numpy as np
import torch.nn.functional as F

from src.utils import DotDict, get_paper_summary_stats_new
from src.model_loader import load_target_model
from src.data_utils import generate_data, load_pickle
from src.strategies.soda import onehot_search
from src.strategies.gcg import gcg_search

def main():
    parser = argparse.ArgumentParser(description="LLM Inversion Experiments")
    parser.add_argument("--model_name", type=str, default="tiny-stories-33M")
    parser.add_argument("--method", type=str, choices=["soda_text", "soda_logits", "gcg_text", "gcg_logits"], required=True)
    parser.add_argument("--input_len", type=int, default=2)
    parser.add_argument("--num_targets", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # Configuration (Defaults from Notebook)
    cfg = DotDict({
        "model_name": args.model_name,
        "input_len": args.input_len,
        "num_targets": args.num_targets,
        "learn_rate": 0.065,
        "decay_rate": 0.9,
        "betas": (0.9, 0.995),
        "temp": 0.05,
        "max_epochs": 200, # Lowered for demo speed
        "save_folder": f"output/{args.method}_{args.model_name}",
        "target_strategy": "random",
        "init_strategy": "zeros",
        "output_len": 25,
        # GCG params
        "adjusted_max_epochs": 200,
        "top_k": 128,
        "num_candidates": 32,
        "num_mutations": 1,
        "max_batch_size": 4,
        "bias_correction": False,
        "reg_weight": None,
        "reset_epoch": 50,
        "reinit_epoch": 1500,
    })
    
    device = args.device
    model = load_target_model(cfg.model_name, device)

    # Check Data
    data_path = f"data/true_tokens_{cfg.num_targets}_{cfg.input_len}.pkl"
    if not os.path.exists(data_path):
        generate_data(model, cfg, device)
    
    # Load Data based on mode
    loaded_true_tokens = load_pickle(data_path, "cpu")
    loaded_true_outputs = None
    loaded_initial_tokens = None
    
    if "text" in args.method:
         greedy_path = f"data/true_tokens_{cfg.num_targets}_{cfg.input_len}_{cfg.output_len}_greedy.pkl"
         if os.path.exists(greedy_path):
             loaded_true_outputs = load_pickle(greedy_path, "cpu")
         else:
             # Regenerate if missing (simplified handling)
             generate_data(model, cfg, device)
             loaded_true_outputs = load_pickle(greedy_path, "cpu")
    
    # Prepare Inits for SODA
    initialisation_embeds = None
    if "soda" in args.method:
        # For 'zeros' strategy
        initialisation_embeds = torch.zeros((cfg.num_targets, cfg.input_len, model.embed.W_E.size(0))).to("cpu")
    
    # Prepare Inits for GCG
    if "gcg" in args.method:
        # For 'zeros' strategy (token ids)
        loaded_initial_tokens = torch.zeros((cfg.num_targets, cfg.input_len), dtype=torch.long).to("cpu")

    # Run Experiment
    print(f"Running {args.method}...")
    results = []
    elapsed = 0
    
    mode = "text" if "text" in args.method else "logits"

    if "soda" in args.method:
        results, elapsed = onehot_search(
            model, cfg, device, mode=mode,
            loaded_true_tokens=loaded_true_tokens,
            loaded_true_outputs=loaded_true_outputs,
            initialisation_embeds=initialisation_embeds
        )
    elif "gcg" in args.method:
        results, elapsed = gcg_search(
            model, cfg, device, mode=mode,
            loaded_true_tokens=loaded_true_tokens,
            loaded_true_outputs=loaded_true_outputs,
            loaded_initial_tokens=loaded_initial_tokens
        )

    # Stats
    stats = get_paper_summary_stats_new(results, epochs=cfg.max_epochs)
    stats["elapsed_time"] = elapsed
    
    os.makedirs(cfg.save_folder, exist_ok=True)
    print(f"\nStats: {json.dumps(stats, indent=2)}")
    
    with open(f'{cfg.save_folder}/stats.json', 'w') as f:
        json.dump(stats, f)
    with open(f'{cfg.save_folder}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()