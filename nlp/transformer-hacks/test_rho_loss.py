#  torchrun --nproc_per_node=2 test_rho_loss.py
#  Optional: RHO_IL_TAG=<tag> torchrun ... test_rho_loss.py  (tests tag mode)
#  Default: tests EMA self-IL (no pretrained model needed)
import os
import torch
import torch.distributed as dist
from datetime import timedelta

from training.model import Model, Config, TransformerLayerType, SamplingMethod
from training.optimizer import AdamWConfig
from training.trainer import TrainingOptions, DistributedStrategy
from training.rho_loss import RhoLossTagConfig, RhoLossEmaConfig, RhoLossSnapshotConfig
from experiments import execute, NAMED_DATASETS

dist.init_process_group("nccl", timeout=timedelta(seconds=300))
rank = dist.get_rank()
torch.cuda.set_device(rank)

dataset = NAMED_DATASETS["fineweb-256"]
config = Config(
    sequence_length=256,
    dim_embeddings=256,
    num_attention_heads=4,
    num_transformer_layers=4,
    padding_index=0,
    vocab_size=dataset.vocab_size,
    transformer_layer=TransformerLayerType.GPT2,
)
config.sampling_method = SamplingMethod.ARGMAX

mode = os.environ.get("RHO_MODE", "ema")
il_tag = os.environ.get("RHO_IL_TAG")
if il_tag:
    rho = RhoLossTagConfig(tag=il_tag, ratio=0.2)
elif mode == "snapshot":
    rho = RhoLossSnapshotConfig(ratio=0.2, snapshot_steps=50, warmup_steps=20)
else:
    rho = RhoLossEmaConfig(ratio=0.2, decay=0.999, warmup_steps=20)

options = TrainingOptions(
    batch_size=32,
    epochs=1,
    training_timeout_minutes=2,
    optimizer=AdamWConfig(lr=3e-4),
    distributed_strategy=DistributedStrategy.FSDP,
    device=torch.device(f"cuda:{rank}"),
    rho_loss=rho,
)

execute(dataset, "rho-loss-test", Model(config), options)
