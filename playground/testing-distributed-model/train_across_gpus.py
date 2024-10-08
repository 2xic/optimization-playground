"""
This trains large model on multiple GPUs

Running the following should work.

torchrun --nproc-per-node 2 train_across_gpus.py

Good references 
- https://github.com/pytorch/PiPPy/blob/main/examples/huggingface/pippy_gpt2.py
- https://huggingface.co/docs/transformers/en/perf_train_gpu_many
"""

from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.distributed.PipelineDistrubted import MultipleGpuBigModelWrapper
from optimization_playground_shared.models.BasicConvModel import BasicConvModel


class Trainer(MultipleGpuBigModelWrapper):
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    train, _ = get_dataloader(
        batch_size=16
    )
    trainer = Trainer()
    trainer.start()
    model = BasicConvModel(input_shape=(1, 28, 28))
    trainer.run(
        model,
        train,
        split_spec=trainer.get_parameters_split_name(model)
    )
