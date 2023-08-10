import torch
import torch
from resnet_adjusted import get_renset_18
import time
from optimization_playground_shared.process_pools.MultipleGpus import run_on_multiple_gpus, ddp_setup
# from optimization_playground_shared.training_loops.TrainingLoopProfile import TrainingLoopProfile
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.distributed.TrainingLoopDistributedAccumulate import TrainingLoopDistributedAccumulate
from optimization_playground_shared.utils.GlobalTimeSpentInFunction import GlobalTimeSpentInFunction
import time
from torch.distributed import destroy_process_group
import torch
from optimization_playground_shared.dataloaders.ResnetCifar10 import get_dataloader
from optimization_playground_shared.utils.Timer import Timer
from torch.utils.data.distributed import DistributedSampler
import json
import sys

torch.backends.cudnn.benchmark = True
# torch.set_default_device("cuda")
# torch.multiprocessing.set_start_method('spawn')

RUN_ON_MULTIPLE_GPUS = False
SIZE = int(sys.argv[1])

"""
-> Maybe split the model over multiple GPUS
    -> https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
    -> https://pytorch.org/docs/stable/pipeline.html
    -> Or not, sounds like it slowdowns things
->
"""

def main(gpu_id, world_size, size):
    ddp_setup(gpu_id, world_size)
    core(
        gpu_id, size=size
    )
    destroy_process_group()


def core(gpu_id, size):
    model = get_renset_18(
        width=size
    )
    data_transform = None
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    train, test, _, _ = get_dataloader(
        shuffle=False,
        sampler=(lambda dataset: DistributedSampler(dataset)) if RUN_ON_MULTIPLE_GPUS else None,
        transforms=data_transform,
        batch_size=512,
        num_workers=8,
    )
    if torch.__version__.startswith("2."):
        torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters())

    trainer = None
    if RUN_ON_MULTIPLE_GPUS:
        trainer = TrainingLoopDistributedAccumulate(
            model,
            optimizer,
            gpu_id=gpu_id
        )
    else:
        trainer = TrainingLoop(
            model,
            optimizer,
        )
    start = time.time()

    train_acc = []
    test_acc = []
    train_x = []
    test_x = []

    for epochs in range(4_000):
        acc = None
        with Timer("train"):
            _, acc = trainer.train(train)
        testing_acc = None
        with Timer("testing"):
            testing_acc = trainer.eval(test)
        train_acc.append(acc.item())
        test_acc.append(testing_acc.item())
        test_x.append(trainer.epoch)

        if epochs % 10 == 0:
            print(f"Epoch: {epochs}, Acc: {acc}, Acc (test): {testing_acc}")
            end = time.time()

            if gpu_id == 0:
                results = {
                    "epoch": epochs,
                    "size": size,
                    "time": (end - start),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "params": pytorch_total_params
                }
                with open(f"{size}.json", "w") as file:
                    json.dump(results, file)

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES= *device id*
    print(f"Training with width={SIZE}")
    if RUN_ON_MULTIPLE_GPUS:
        out = run_on_multiple_gpus(main, SIZE)
    else:
        out = core(0, SIZE)
