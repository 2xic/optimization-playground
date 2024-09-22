"""
This does Pipeline distribution, allowing bigger models to be trained across 
multiple GPUs
"""
from torch.distributed.pipelining import ScheduleGPipe
from torch.distributed.pipelining import pipeline, SplitPoint, Pipe
import torch
import abc
import torch.distributed as dist
import os
from ..utils.GetParameterCoumt import get_parameter_count
import os
import time
from tqdm import tqdm

class MultipleGpuBigModelWrapper(abc.ABC):
    def start(self) -> None:
        self.rank = int(os.getenv("RANK", -1))
        self.world_size = int(os.getenv("WORLD_SIZE", 4))
        self.chunks = self.world_size
        self._main()


        self.losses = []
        self.train = True

    def get_parameters_split_name(self, module: torch.nn.Module):
        parameters = module.named_modules()
        # _ = next(parameters) # first parameter, we skip as pytorch want the next one
        first_trainable_parameter = None
        for index, (name, module) in enumerate(parameters):
            if index > 1:
                first_trainable_parameter = name # value._get_name()
                break
        return {
            str(first_trainable_parameter): SplitPoint.BEGINNING,
        }

    def setup(self, module: torch.nn.Module, dataloader, split_spec):
        X, y = next(iter(dataloader))

        X = X.to(self.device)
        y = y.to(self.device)

        module.to(self.device)
        module.eval()

       # print((next(module.parameters()).device, X.device))

        input_X = X[:X.shape[0] // self.chunks].reshape((X.shape[0] // self.chunks, ) + tuple(X.shape[1:]))
        input_Y = y[:y.shape[0] // self.chunks].reshape((y.shape[0] // self.chunks, ) + tuple(y.shape[1:]))
        assert input_X.shape[0] == X.shape[0] // self.chunks
        assert input_Y.shape[0] == y.shape[0] // self.chunks

      #  print((first_trainable_parameter, split_start))
        pipe: Pipe = pipeline(
            module=module,
            mb_args=(),
            mb_kwargs={
                "x": input_X,
            },
            split_spec=split_spec
        )
        smod = pipe.get_stage_module(self.rank)
      #  print(f"Pipeline stage {self.rank} {get_parameter_count(smod) / 10 ** 6}M params")

        self.stage = pipe.build_stage(
            self.rank,
            device=self.device,
        )

        self.optimizer = torch.optim.Adam(smod.parameters())

        # Attach to a schedule
        self.schedule = ScheduleGPipe(
            self.stage, 
            self.chunks,
            loss_fn=(torch.nn.functional.cross_entropy if self.train else None)
        )

        # turn training back on so we can train this model ... 
        smod.train()

    def run_epoch(self, dataloader, epochs, view_function=lambda x: x):
        for epoch in range(epochs):
            dataloader_iterator = tqdm(dataloader) if self.rank == 0 else dataloader
            for _, (X, y) in enumerate(dataloader_iterator):
                X = X.to(self.device)
                y = y.to(self.device)

                # TODO: Figure out why it doesn't work without the batch size
                if X.shape[0] != dataloader.batch_size:
                    break

                assert X.shape[0] == y.shape[0]

                out = self.forward(X, y, view_function)
                if out is not None:
                    self.batch_done(self.losses, view_function(y), out)

                self.optimizer.step()
                dist.barrier()

            if self.stage.is_last:
                self.epoch_done(epoch)

        dist.destroy_process_group()

    def forward(self, X, y=None, view_function=lambda x: x):
        train = y != None
        if self.rank == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.schedule.step(
                x=X,
                target=(view_function(y) if train else None),
                losses=(self.losses if train else None),
            )
        else:
            out = self.schedule.step(
                target=(view_function(y) if train else None),
                losses=(self.losses if train else None),
            )
            if self.stage.is_last:
            #  assert out.shape[0] == X.shape[0], f"Mismatch with view input {X.shape} / {out.shape[0]}"
                assert out.shape[0] == view_function(y).shape[0], f"Mismatch with view function {out.shape} / {view_function(y).shape} and X ({X.shape})"
                return out
        return None
        
    def _main(self):
        if not "RANK" in os.environ:
            print(f"Not executed correctly. Instead run: ")
            print(f"torchrun --nproc-per-node {torch.cuda.device_count()} [file.py]")
            exit(0)

        dev_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{dev_id}")

        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device,
        )

    def batch_done(self, losses, y: torch.Tensor , y_predicted: torch.Tensor):
        loss = sum([i.item() for i in losses])
        print(f"Loss {loss}")

    def epoch_done(self, epoch):
        pass
