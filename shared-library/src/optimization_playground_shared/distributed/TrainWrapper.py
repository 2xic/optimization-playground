"""
All the setup code you need to start training on multiple GPUs
"""
from ..process_pools.MultipleGpus import run_on_multiple_gpus, ddp_setup
from torch.distributed import destroy_process_group
from .TrainingLoopDistributed import TrainingLoopDistributed
import abc
import torch

class MultipleGpuTrainWrapper(abc.ABC):   
    def start(self) -> None:
        run_on_multiple_gpus(self._main)

    def _main(self, gpu_id, world_size):
        ddp_setup(gpu_id, world_size)
        self._core(
            gpu_id
        )
        destroy_process_group()

    def _core(self, gpu_id):
        model, optimizer = self._get_model_and_optimizer()
        trainer = TrainingLoopDistributed(
            model=model,
            optimizer=optimizer,
            gpu_id=gpu_id,
            loss=self._loss()
        )
        dataloader = self._get_dataloader(gpu_id)
        for epoch in range(1_000):
            results = trainer.train(dataloader)
            if results is not None:
                (loss, accuracy) = results
                self._epoch_done(epoch, model, loss, accuracy, gpu_id)

    @abc.abstractmethod
    def _get_model_and_optimizer(self):
        pass

    @abc.abstractmethod
    def _get_dataloader(self, device):
        pass

    def _loss(self):
        return torch.nn.CrossEntropyLoss()
    
    def _epoch_done(self, epoch, model, loss, accuracy, device):
        pass
