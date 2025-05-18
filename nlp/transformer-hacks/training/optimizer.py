import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, factor=1.0, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        scale = (self.d_model**-0.5) * min(step**-0.5, step * (self.warmup_steps**-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


def lr_lambda(step):
    if step < 100:
        return 1
    elif step < 200:
        return 1e-1
    elif step < 1_000:
        return 1e-2
    else:
        return 1e-4


BETA_1 = 0.90
BETA_2 = 0.95


class AdamOptimizerWrapper(torch.optim.Adam):
    def __init__(
        self,
        params,
        lr=3e-4,
        max_grad_norm=None,
    ):
        super().__init__(
            params,
            lr,
            betas=(
                BETA_1,
                BETA_2,
            ),
            weight_decay=1e-1,
        )
        self.params = params
        self.defaults["max_grad_norm"] = max_grad_norm

    def step(self, closure=None):
        max_grad_norm = self.defaults.get("max_grad_norm")
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.params, max_grad_norm)
        return super().step(closure)
