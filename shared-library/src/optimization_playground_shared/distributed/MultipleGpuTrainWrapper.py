"""
All the setup code you need to start training on multiple GPUs
"""
from ..process_pools.MultipleGpus import run_on_multiple_gpus, ddp_setup
from .MultipleGPUsTrainingLoopDistributedAccumulate import MultipleGPUsTrainingLoopDistributedAccumulate
from ..training_loops.TrainingLoop import TrainingLoop
import abc

"""
This does distributed data training
"""
class MultipleGpuTrainWrapper(abc.ABC):
    def start(self, is_debug_mode=False) -> None:
        if not is_debug_mode:
            run_on_multiple_gpus(self._main, is_debug_mode)
        else:
            self._main(gpu_id=0, world_size=None, is_debug_mode=is_debug_mode)

    def _main(self, gpu_id, world_size, is_debug_mode):
        if not is_debug_mode:
            ddp_setup(gpu_id, world_size)
        print(f"is_debug_mode : {is_debug_mode}, gpu_id: {gpu_id}")
        self._core(
            gpu_id,
            is_debug_mode
        )

    def _core(self, gpu_id, is_debug_mode):
        # Get dataloader first to init global variables
        dataloader = self.get_dataloader(gpu_id)
        model, optimizer, loss = self.get_training_parameters()
        trainer = None 
        
        # if not debug mode use the distributed version, else test on single GPU
        if not is_debug_mode:
            trainer = MultipleGPUsTrainingLoopDistributedAccumulate(
                model=model,
                optimizer=optimizer,
                gpu_id=gpu_id,
                loss=loss,
            )
        else:
            trainer = TrainingLoop(
                model=model,
                optimizer=optimizer,
                loss=loss,
            )
        print(f"Starting to train on {gpu_id}")
        for epoch in range(1_000):
            results = trainer.train(dataloader)
            if results is not None:
                (loss, accuracy) = results
                print(f"loss: {loss}, accuracy: {accuracy}")
                self.epoch_done(epoch, model, loss, accuracy, gpu_id)

    @abc.abstractmethod
    def get_training_parameters(self):
        pass

    @abc.abstractmethod
    def get_dataloader(self, device):
        pass
    
    def epoch_done(self, epoch, model, loss, accuracy, device):
        pass
