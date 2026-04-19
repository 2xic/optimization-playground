#  torchrun --nproc_per_node=2 test_muon.py
import torch
import torch.distributed as dist
from datetime import timedelta

from training.model import Model, Config, TransformerLayerType, SamplingMethod
from training.optimizer import MuonConfig
from training.trainer import TrainingOptions, DistributedStrategy
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

options = TrainingOptions(
    batch_size=32,
    epochs=1,
    training_timeout_minutes=2,
    optimizer=MuonConfig(lr=3e-4),
    distributed_strategy=DistributedStrategy.FSDP,
    device=torch.device(f"cuda:{rank}"),
)

execute(dataset, "muon-test", Model(config), options)
