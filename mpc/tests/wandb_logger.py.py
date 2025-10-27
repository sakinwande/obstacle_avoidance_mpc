# wandb_logger.py (or stick this near the top of your script)
import time, wandb, torch, numpy as np

class WandbLogger:
    def __init__(self, project="il_from_mpc", run_name=None, config=None):
        self.run = wandb.init(project=project, name=run_name, config=config or {}, save_code=True)
        self.t0 = time.time()

    def log_train(self, *, epoch, loss, lr=None, grad_norm=None, n_seen=None):
        payload = {"train/loss": float(loss), "epoch": int(epoch)}
        if lr is not None:        payload["train/lr"] = float(lr)
        if grad_norm is not None: payload["train/grad_norm"] = float(grad_norm)
        if n_seen is not None:    payload["data/examples_seen"] = int(n_seen)
        payload["meta/elapsed_s"] = time.time() - self.t0
        wandb.log(payload, step=epoch)

    def log_eval_agreement(self, *, epoch, student_fn, expert_fn, sampler, n=1024):
        with torch.no_grad():
            S = sampler(n)                       # (n, state_dim) np.float32
            S_t = torch.from_numpy(S)
            a_student = student_fn(S_t).cpu().numpy()
            a_expert  = np.stack([expert_fn(S_t[i]).numpy() for i in range(n)], axis=0)
            l2 = np.linalg.norm(a_student - a_expert, axis=1)
            metrics = {
                "eval/mean_l2": float(l2.mean()),
                "eval/p95_l2":  float(np.percentile(l2, 95)),
                "eval/max_l2":  float(l2.max()),
                "epoch": int(epoch),
            }
        wandb.log(metrics, step=epoch)

    def finish(self):
        wandb.finish()

def grad_global_norm(model):
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            tot += float(torch.sum(g*g))
    return float(tot**0.5)
