import torch
from torch.optim.lr_scheduler import _LRScheduler
from dataclasses import dataclass

import torch.optim.rmsprop
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def create_optimizer(self, params):
        pass


class Scheduler(ABC):
    @abstractmethod
    def create_scheduler(self, optimizer):
        pass


class NoamScheduler(_LRScheduler):
    def __init__(self, d_model, warmup_steps, factor=1.0, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.last_epoch = last_epoch
        self.factor = factor

    def create_scheduler(self, optimizer):
        super(NoamScheduler, self).__init__(optimizer, self.last_epoch)

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


@dataclass
class AdamConfig:
    lr: float = 3e-4
    max_grad_norm: float = 0
    betas: tuple = (BETA_1, BETA_2)
    eps: float = 1e-8
    weight_decay: float = 0

    def create_optimizer(self, params):
        return AdamOptimizerWrapper(
            params,
            max_grad_norm=self.max_grad_norm,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            eps=self.eps,
            fused=True,
        )


@dataclass
class RMSpropConfig:
    lr: float = 3e-4
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0
    momentum: float = 0

    def create_optimizer(self, params):
        return torch.optim.RMSprop(
            params,
            lr=self.lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )


class AdamOptimizerWrapper(torch.optim.Adam):
    def __init__(
        self,
        params,
        max_grad_norm,
        **kwargs,
    ):
        super().__init__(
            params,
            **kwargs,
        )
        self.params = params
        self.defaults["max_grad_norm"] = max_grad_norm

    def step(self, closure=None):
        max_grad_norm = self.defaults.get("max_grad_norm")
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.params, max_grad_norm)
        return super().step(closure)
