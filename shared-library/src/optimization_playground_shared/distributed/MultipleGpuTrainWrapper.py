"""
All the setup code you need to start training on multiple GPUs
"""
import torch.utils
import torch.utils.data
from ..process_pools.MultipleGpus import run_on_multiple_gpus, ddp_setup
from .MultipleGPUsTrainingLoopDistributedAccumulate import MultipleGPUsTrainingLoopDistributedAccumulate
from ..training_loops.TrainingLoop import TrainingLoop
import abc
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from dataclasses import dataclass
import torch

@dataclass
class ModelParameters:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss: torch.nn.Module
    dataloader: torch.utils.data.DataLoader

"""
This does distributed data training
"""
class MultipleGpuTrainWrapper(abc.ABC):
    def __init__(self):
        super().__init__()
        self.is_debug_mode = not torch.cuda.is_available()
        self.epochs = 1_000

    def start(self, is_debug_mode=False) -> None:
        if not is_debug_mode:
            run_on_multiple_gpus(self._main, is_debug_mode)
        else:
            print("Running in debug mode")
            self._main(gpu_id=0, world_size=None, is_debug_mode=is_debug_mode)

    def _main(self, gpu_id, world_size, is_debug_mode):
        if not is_debug_mode:
            ddp_setup(gpu_id, world_size)
        print(f"is_debug_mode : {is_debug_mode}, gpu_id: {gpu_id}")
        self._core(gpu_id)
        self.is_debug_mode = is_debug_mode

    def _core(self, gpu_id):
        # Get dataloader first to init global variables
        output = self.get_training_parameters()
        self.model = output.model
        self.optimizer= output.optimizer
        self.loss = output.optimizer
        self.dataloader = output.dataloader
        if not self.is_debug_mode:
            self.model = DDP(self.model.to(gpu_id), device_ids=[gpu_id])
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.trainer = None
        self.train(self.device)
        
    def train(self, _device: torch.device):
        # if is debug mode use the distributed version, else test on single GPU
        if self.is_debug_mode:
            trainer = TrainingLoop(
                model=self.model,
                optimizer=self.optimizer,
                loss=self.loss,
            )
        else:
            trainer = MultipleGPUsTrainingLoopDistributedAccumulate(
                model=self.model,
                optimizer=self.optimizer,
                gpu_id=self.gpu_id,
                loss=self.loss,
            )
        print(f"Starting to train on {self.gpu_id}")
        for epoch in range(self.epochs):
            trainer._batch_done = self.batch_done
            results = trainer.train(self.dataloader)
            if results is not None:
                (loss, accuracy) = results
                print(f"loss: {loss}, accuracy: {accuracy}")
                self.epoch_done(epoch, self.model, loss, accuracy, self.gpu_id)

    @abc.abstractmethod
    def get_training_parameters(self) -> ModelParameters:
        pass
    
    def epoch_done(self, epoch, model, loss, accuracy, device):
        pass

    def batch_done(self):
        pass
