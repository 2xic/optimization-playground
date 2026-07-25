import copy
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union
from contextlib import nullcontext
from torch.amp import autocast

from utils.load_mode_from_checkpoint import load_modeL_tag, load_model_from_path


@dataclass
class RhoLossTagConfig:
    tag: str
    ratio: float = 0.2


@dataclass
class RhoLossEmaConfig:
    ratio: float = 0.2
    decay: float = 0.999
    warmup_steps: int = 100


@dataclass
class RhoLossSnapshotConfig:
    ratio: float = 0.2
    snapshot_steps: int = 500
    warmup_steps: int = 100


RhoLossConfig = Union[RhoLossTagConfig, RhoLossEmaConfig, RhoLossSnapshotConfig]


def _load_il_from_tag(tag: str, device: torch.device, dtype: Optional[torch.dtype] = None) -> nn.Module:
    path = load_modeL_tag(tag)
    model, _config = load_model_from_path(path)
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def _clone_frozen(main_model: nn.Module, dtype: Optional[torch.dtype] = None) -> nn.Module:
    clone = copy.deepcopy(main_model)
    if dtype is not None:
        clone = clone.to(dtype)
    clone.eval()
    clone.requires_grad_(False)
    return clone


@dataclass
class RhoLossSelector:
    il_model: nn.Module
    config: RhoLossConfig
    device: torch.device
    dtype: Optional[torch.dtype] = None
    log_first_steps: int = 3
    _step: int = 0

    @classmethod
    def build(cls, config: RhoLossConfig, main_model: nn.Module, device: torch.device, dtype: Optional[torch.dtype] = None) -> "RhoLossSelector":
        if isinstance(config, RhoLossTagConfig):
            il = _load_il_from_tag(config.tag, device, dtype)
        elif isinstance(config, (RhoLossEmaConfig, RhoLossSnapshotConfig)):
            il = _clone_frozen(main_model, dtype)
        else:
            raise TypeError(f"Unknown RhoLossConfig: {type(config)}")
        return cls(il_model=il, config=config, device=device, dtype=dtype)

    @property
    def mode(self) -> str:
        if isinstance(self.config, RhoLossTagConfig):
            return "tag"
        if isinstance(self.config, RhoLossEmaConfig):
            return "ema"
        return "snapshot"

    @property
    def ratio(self) -> float:
        return self.config.ratio

    def select(self, X: torch.Tensor, y: torch.Tensor, main_model: nn.Module, objective):
        B = X.shape[0]
        k = max(1, round(B * self.ratio))
        warmup = getattr(self.config, "warmup_steps", 0)
        if k >= B or self._step < warmup:
            self._step += 1
            return X, y
        ctx = autocast("cuda", dtype=self.dtype) if self.dtype is not None else nullcontext()
        was_training = main_model.training
        main_model.eval()
        try:
            with torch.no_grad(), ctx:
                il_logits = self.il_model(X)
                main_logits = main_model(X)
                l_il = objective.per_sequence_loss(il_logits, y)
                l_main = objective.per_sequence_loss(main_logits, y)
        finally:
            if was_training:
                main_model.train()
        scores = (l_main - l_il).float()
        idx = torch.topk(scores, k=k).indices
        if self._step < self.log_first_steps:
            try:
                import torch.distributed as dist
                rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                rank0 = True
            if rank0:
                print(f"[rho-loss][{self.mode}] step={self._step} B={B} k={k} score_mean={scores.mean().item():.4f} score_max={scores.max().item():.4f}", flush=True)
        self._step += 1
        return X[idx], y[idx]

    @torch.no_grad()
    def update(self, main_model: nn.Module):
        if isinstance(self.config, RhoLossEmaConfig):
            self._copy_into_il(main_model, alpha=1.0 - self.config.decay, decay=self.config.decay)
        elif isinstance(self.config, RhoLossSnapshotConfig) and self._step > 0 and self._step % self.config.snapshot_steps == 0:
            self._copy_into_il(main_model, alpha=1.0, decay=0.0)

    def _copy_into_il(self, main_model: nn.Module, alpha: float, decay: float):
        il_params = dict(self.il_model.named_parameters())
        il_buffers = dict(self.il_model.named_buffers())
        for name, p in main_model.named_parameters():
            if name not in il_params:
                continue
            src = p.detach()
            if il_params[name].dtype != src.dtype:
                src = src.to(il_params[name].dtype)
            if decay == 0.0:
                il_params[name].copy_(src)
            else:
                il_params[name].mul_(decay).add_(src, alpha=alpha)
        for name, b in main_model.named_buffers():
            if name in il_buffers and il_buffers[name].dtype.is_floating_point:
                src = b.detach()
                if il_buffers[name].dtype != src.dtype:
                    src = src.to(il_buffers[name].dtype)
                il_buffers[name].copy_(src)
