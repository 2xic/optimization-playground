import torch
import torch
from resnet_adjusted import get_renset_18
import time
from optimization_playground_shared.process_pools.MultipleGpus import run_on_multiple_gpus, ddp_setup
#from optimization_playground_shared.distributed.TrainingLoopDistributed import TrainingLoopDistributed
from optimization_playground_shared.distributed.TrainingLoopDistributedAccumulate import TrainingLoopDistributedAccumulate
import time
from torch.distributed import destroy_process_group
import torch
from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader
from optimization_playground_shared.utils.Timer import Timer
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import json

torch.backends.cudnn.benchmark = True
#torch.set_default_device("cuda")

def main(gpu_id, world_size, size):
    ddp_setup(gpu_id, world_size)
    model = get_renset_18(
        width=size
    )
    data_transform = Compose([Resize((224, 224)), ToTensor()])
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    train, test = get_dataloader(
        shuffle=False,
        sampler=lambda dataset: DistributedSampler(dataset),
        transforms=data_transform,
        batch_size=64,
        num_workers=6,
    )
    torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters())
    training = []
    testing = []
    trainer = TrainingLoopDistributedAccumulate(
        model,
        optimizer,
        gpu_id=gpu_id
    )
    start = time.time()

    train_acc = []
    test_acc = []
    train_x = []
    test_x = []
    
    while trainer.epoch < 4_000:
        acc = None
        with Timer("train"):
            _, acc = trainer.train(train)
        testing_acc = None
        with Timer("testing"):
            testing_acc = trainer.eval(test)
        if gpu_id == 0:
            print(f"{acc}")
            print(f"{testing_acc}")
            print(trainer.epoch)
            train_acc.append(acc.item())
            test_acc.append(testing_acc.item())
            test_x.append(trainer.epoch)
            break
        else:
            print(f"Epoch {trainer.epoch}")
            break
    destroy_process_group()

    end = time.time()

    if gpu_id == 0:
        results = {
            "size":size,
            "time":(end - start),
            "train_acc": train_acc,
            "test_acc": test_acc,
            "params": pytorch_total_params
        }
        with open(f"{size}.json", "w") as file:
            json.dump(results, file)

if __name__ == "__main__":
    for i in [
            8,
            16,
            28,
            32,
            48,
            64,
        ]:
        print(f"{i}")
        out = run_on_multiple_gpus(main, i)
        print(out)
        break
