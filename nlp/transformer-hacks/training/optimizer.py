import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ExponentialLR
from dataclasses import dataclass

import torch.optim.rmsprop
from abc import ABC, abstractmethod


class Optimizer(ABC):
    lr: float = 3e-4

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


class ExponentialLR(ExponentialLR):
    def __init__(self, gamma):
        self.gamma = gamma

    def create_scheduler(self, optimizer):
        super(ExponentialLR, self).__init__(optimizer, gamma=self.gamma)


class WarmupExpDecay(_LRScheduler):
    def __init__(self, warmup_epochs=5, gamma=0.95, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        self.last_epoch = last_epoch

    def create_scheduler(self, optimizer):
        super().__init__(optimizer, self.last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            decay_epochs = self.last_epoch - self.warmup_epochs
            return [base_lr * (self.gamma**decay_epochs) for base_lr in self.base_lrs]


def lr_lambda(step):
    if step < 100:
        return 1
    elif step < 200:
        return 1e-1
    elif step < 1_000:
        return 1e-2
    else:
        return 1e-4


@dataclass
class MuonConfig:
    lr: float = 3e-4

    def create_optimizer(self, params):
        try:
            return torch.optim.Muon(
                params,
                lr=self.lr,
            )
        except Exception as e:
            print(e)
            # pip install git+https://github.com/KellerJordan/Muon
            from muon import SingleDeviceMuon

            params = list(params)

            hidden_weights = [
                dict(
                    params=[p for p in params if p.ndim >= 2],
                    use_muon=True,
                    lr=0.02,
                    weight_decay=0.01,
                ),
                # dict(
                #    params=[p for p in params if p.ndim < 2],
                #    use_muon=False,
                #    lr=0.02,
                #    weight_decay=0.01,
                # ),
            ]

            return SingleDeviceMuon(
                hidden_weights,
                lr=self.lr,
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
        self.defaults["max_grad_norm"] = max_grad_norm

    def step(self, closure=None):
        max_grad_norm = self.defaults.get("max_grad_norm")
        if max_grad_norm is not None and max_grad_norm > 0:
            all_params = [p for group in self.param_groups for p in group['params']]
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        return super().step(closure)


@dataclass
class AdamWConfig:
    lr: float = 3e-4
    max_grad_norm: float = 0
    betas: tuple = (BETA_1, BETA_2)
    eps: float = 1e-8
    weight_decay: float = 0.01

    def create_optimizer(self, params):
        return AdamWOptimizerWrapper(
            params,
            max_grad_norm=self.max_grad_norm,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            eps=self.eps,
            fused=True,
        )


class AdamWOptimizerWrapper(torch.optim.AdamW):
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
        self.defaults["max_grad_norm"] = max_grad_norm

    def step(self, closure=None):
        max_grad_norm = self.defaults.get("max_grad_norm")
        if max_grad_norm and max_grad_norm > 0:
            all_params = [p for group in self.param_groups for p in group['params']]
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        return super().step(closure)
