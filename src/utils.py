import torch
import torch.optim
from tqdm import tqdm

class DotDict(dict):
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]

class CustomAdam(torch.optim.Optimizer):
    """
    Extracted directly from the Notebook.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                m_hat = m 
                v_hat = v 
                denom = v_hat.sqrt().add(group['eps'])
                p.data.addcdiv_(m_hat, denom, value=-group['lr'])

        return loss

def get_paper_summary_stats_new(results, epochs):
    stats = {}
    percent_zero_loss = 0
    percent_exact_inversion = 0
    end_epoch = []
    zero_losses_at_epoch = []

    for result in results:
        if result["found_solution"]:
            percent_zero_loss += 1
        if torch.equal(result["true_tokens"], result["pred_tokens"]):
            percent_exact_inversion += 1
        end_epoch.append(result["done_epochs"])

    for e in range(1, epochs):
        if len(zero_losses_at_epoch) == 0:
            current = 0
        else:
            current = zero_losses_at_epoch[-1]
        current += end_epoch.count(e)
        zero_losses_at_epoch.append(current)

    stats["percent_zero_loss"] = round((percent_zero_loss/len(results))*100, 4)
    stats["percent_exact_inversion"] = round((percent_exact_inversion/len(results))*100, 4)
    stats["zero_losses_at_epoch"] = zero_losses_at_epoch

    # Basic success per position stats (simplified from notebook for portability)
    if results:
        input_len = len(results[0]["true_tokens"])
        success_final_epoch = [0 for _ in range(input_len)]
        for i in range(input_len):
            for result in results:
                if result["true_tokens"][i] == result["pred_tokens"][i]:
                    success_final_epoch[i] += 1
            success_final_epoch[i] = round(success_final_epoch[i]/len(results)*100, 4)
        stats["success_final_epoch"] = success_final_epoch

    return stats