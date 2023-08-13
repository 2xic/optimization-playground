"""
All the setup code you need to start training on multiple GPUs
"""
from ..process_pools.MultipleGpus import run_on_multiple_gpus, ddp_setup
from .TrainingLoopDistributedAccumulate import TrainingLoopDistributedAccumulate
from ..training_loops.TrainingLoop import TrainingLoop
import abc
import torch

class MultipleGpuTrainWrapper(abc.ABC):
    def start(self, is_debug_mode=False) -> None:
        if not is_debug_mode:
            run_on_multiple_gpus(self._main, is_debug_mode)
        else:
            self._main(gpu_id=0, world_size=None, is_debug_mode=is_debug_mode)

    def _main(self, gpu_id, world_size, is_debug_mode):
        if not is_debug_mode:
            ddp_setup(gpu_id, world_size)
        self._core(
            gpu_id,
            is_debug_mode
        )

    def _core(self, gpu_id, is_debug_mode):
        # Get dataloader first to init global variables
        dataloader = self._get_dataloader(gpu_id, is_debug_mode)
        model, optimizer = self._get_model_and_optimizer()
        trainer = None 
        
        if not is_debug_mode:
            trainer = TrainingLoopDistributedAccumulate(
                model=model,
                optimizer=optimizer,
                gpu_id=gpu_id,
                loss=self._loss()
            )
        else:
            trainer = TrainingLoop(
                model=model,
                optimizer=optimizer,
            )
        print(f"Starting to train on {gpu_id}")
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
