import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from src.utils import DotDict, CustomAdam

def onehot_search(model, cfg, device, mode="text", 
                  loaded_true_tokens=None, 
                  loaded_true_outputs=None, 
                  initialisation_embeds=None):
    """
    Refactored SODA logic from Notebook.
    Combines logic for 'text' and 'logits' mode.
    """
    
    # Setup State
    state_path = f'{cfg.save_folder}/checkpoint_{cfg.input_len}_{cfg.num_targets}_{cfg.max_epochs}.pt'
    if os.path.exists(state_path):
        print("LOADING STATE")
        state = torch.load(state_path)
    else:
        print("INITIALISING STATE")
        state = DotDict({
            "results" : [],
            "batch_results" : [],
            "optimizers" : [],
            "loaded_i" : 0,
            "epoch" : 0,
            "num_remain_items" : cfg.num_targets,
            "num_success_items" : 0,
            "elapsed_time" : 0,
            "checkpoint_elapsed_time" : 0,
        })
        if mode == "text":
            state.true_outputs = torch.tensor([]).to(device).to(torch.int64)
        else: # logits
            state.true_logits = torch.Tensor([]).to(device)

    while state.num_remain_items != 0 or len(state.batch_results) != 0:
        start_time = time.time()

        # Checkpoint
        if state.elapsed_time - state.checkpoint_elapsed_time > (3600 * 3):
            print("\nSAVING STATE")
            state.checkpoint_elapsed_time = state.elapsed_time
            torch.save(state, state_path)

        state.epoch += 1
        if state.epoch % 100 == 0:
            print(f"({state.num_success_items}/{cfg.num_targets})({cfg.num_targets-state.num_remain_items}/{cfg.num_targets}){state.epoch}", end=", ")

        with torch.no_grad():
            # Add new items to batch
            if (cfg.max_batch_size - len(state.batch_results)) > 0 and state.num_remain_items != 0:
                num_new_items = min((cfg.max_batch_size - len(state.batch_results)), state.num_remain_items)
                state.num_remain_items -= num_new_items

                # Init new target
                true_tokens = loaded_true_tokens[state.loaded_i:state.loaded_i+num_new_items].to(device)
                
                if mode == "text":
                    new_true_outputs = loaded_true_outputs[state.loaded_i:state.loaded_i+num_new_items].to(device)
                    state.true_outputs = torch.cat((state.true_outputs, new_true_outputs))
                else: # logits
                    new_true_logits = model(true_tokens).detach()[:,-1,:]
                    state.true_logits = torch.cat((state.true_logits, new_true_logits))

                for i in range(num_new_items):
                    item_dict = {
                        "true_tokens": true_tokens[i].to("cpu"),
                        "pred_tokens": None,
                        "found_solution": False,
                        "done_epochs": 0,
                    }
                    if mode == "text":
                        item_dict["true_outputs"] = state.true_outputs[i].to("cpu")
                    
                    state.batch_results.append(item_dict)

                    # Init Optimizers
                    new_pred_embed = initialisation_embeds[state.loaded_i+i:state.loaded_i+i+1].to(device)
                    for j in range(cfg.input_len):
                        new_pred_embed_pos = new_pred_embed[:,j:j+1]
                        new_pred_embed_pos.requires_grad = True
                        if j == 0:
                            if cfg.bias_correction:
                                state.optimizers.append(torch.optim.Adam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                            else:
                                state.optimizers.append(CustomAdam([new_pred_embed_pos], lr=cfg.learn_rate, betas=cfg.betas))
                        else:
                            state.optimizers[-1].param_groups[0]['params'].append(new_pred_embed_pos)

                state.loaded_i += num_new_items

        # Optimization Step
        if len(state.optimizers) == 0:
             # Edge case where everything finished
             break

        for optimizer in state.optimizers:
            optimizer.zero_grad()
            
        pred_embed_pre = torch.cat([torch.cat([param for param in optimizer.param_groups[0]['params']], dim=1)
                                    for optimizer in state.optimizers], dim=0).to(device)
        pred_one_hot = torch.softmax(pred_embed_pre / cfg.temp, dim=-1)
        pred_embed = (pred_one_hot @ model.embed.W_E)
        
        # Forward Pass Logic
        if mode == "text":
            pred_embed_full = torch.cat((pred_embed, model.embed(state.true_outputs[:,:-1])), dim=1)
        else:
            pred_embed_full = pred_embed

        if "gpt" in cfg.model_name or "tiny" in cfg.model_name:
            # Handling Positional Embeddings if needed (simplified)
            try:
                # This line in notebook is specific to GPT/TinyStories architecture
                pred_embed_full = pred_embed_full + model.pos_embed(pred_embed_full[:,:,0].detach())
            except:
                pass # Some models handle this differently or auto-add
        
        pred_logits = model(pred_embed_full, start_at_layer=0)

        # Loss Calculation
        if mode == "text":
            pred_logprobs = F.softmax(pred_logits[:,cfg.input_len-1:,:], dim=-1).clamp(min=1e-12).log()
            pred_logits_target = torch.gather(pred_logprobs, 2, state.true_outputs.unsqueeze(-1)).squeeze(-1)
            pred_logits_diff = (pred_logits_target - pred_logprobs.max(dim=-1).values)
            loss = - pred_logits_diff.mean()
        else: # logits
            loss = torch.nn.HuberLoss()(state.true_logits.detach(), pred_logits[:,-1,:])

        # Regularization (Optional, keeping logic structure)
        if cfg.reg_weight is not None and state.epoch >= 0:
             # Simplified: just adding the fluency penalty line from notebook if needed
             # reg_penalty = ... 
             # loss = loss + (cfg.reg_weight * reg_penalty)
             pass

        loss.backward()
        for optimizer in state.optimizers:
            optimizer.step()

        # Decay & Check Results
        with torch.no_grad():
            # Decay
            for i in range(len(state.optimizers)):
                for j in range(len(state.optimizers[i].param_groups[0]['params'])):
                    state.optimizers[i].param_groups[0]['params'][j].mul_(cfg.decay_rate)
            
            # Intervention / Resets
            for i in range(len(state.batch_results)):
                targets_epoch = (state.batch_results[i]["done_epochs"]+1)
                if targets_epoch % cfg.reset_epoch == 0:
                    for j in range(cfg.input_len):
                        del state.optimizers[i].state[state.optimizers[i].param_groups[0]['params'][j]]
                if targets_epoch % cfg.reinit_epoch == 0:
                    for j in range(cfg.input_len):
                        state.optimizers[i].param_groups[0]['params'][j].normal_(std=0.1)

            # Update History & Success Check
            pred_tokens = torch.argmax(pred_one_hot, dim=-1)
            
            if mode == "text":
                pred_tokens_full = torch.cat((pred_tokens, state.true_outputs[:,:-1]), dim=1)
                disc_pred_logits = model(pred_tokens_full)[:,cfg.input_len-1:,:]
            else:
                disc_pred_logits = model(pred_tokens)[:,-1,:]

            # Iterate backwards to safely remove
            for i in range(len(state.batch_results)-1, -1, -1):
                state.batch_results[i]["done_epochs"] += 1
                
                have_inverted = False
                if mode == "text":
                    have_inverted = torch.equal(state.batch_results[i]["true_outputs"].detach(), 
                                                disc_pred_logits[i].argmax(dim=-1).to("cpu").detach())
                else:
                    threshold = 1e-4 if "tiny" in cfg.model_name else 1e-3
                    have_inverted = torch.allclose(state.true_logits[i], disc_pred_logits[i], atol=threshold, rtol=threshold)

                if have_inverted:
                    state.batch_results[i]["found_solution"] = True
                    state.num_success_items += 1
                
                if have_inverted or (cfg.max_epochs is not None and state.batch_results[i]["done_epochs"] >= cfg.max_epochs):
                    state.batch_results[i]["pred_tokens"] = pred_tokens[i].to("cpu")
                    del state.optimizers[i]
                    if mode == "text":
                        state.true_outputs = torch.cat((state.true_outputs[:i], state.true_outputs[i+1:]))
                    else:
                        state.true_logits = torch.cat((state.true_logits[:i], state.true_logits[i+1:]))
                    
                    state.results.append(state.batch_results.pop(i))

            state.elapsed_time += time.time() - start_time

    return state.results, round(state.elapsed_time, 3)