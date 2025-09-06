import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from experiments import (
    create_default_config,
    TransformerLayerType,
    Experiment,
    Datasets,
    TrainingOptions,
)
from training.model import Model
from experiments import create_next_token_prediction_objective, AdamConfig


def example(rank, world_size):
    dataset = Datasets.tiny_evm_bytecode()
    experiment = Experiment(dataset)
    experiment.skip_thread = True

    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model_config = create_default_config(
        dataset,
    ).with_transformer_layer(TransformerLayerType.BERT)

    model = Model(model_config).to(rank)

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])

    trainer = create_next_token_prediction_objective(dataset, ddp_model, AdamConfig())
    trainer.train(dataset, TrainingOptions(device=torch.device(rank)))


def main():
    world_size = 2
    mp.spawn(
        example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
