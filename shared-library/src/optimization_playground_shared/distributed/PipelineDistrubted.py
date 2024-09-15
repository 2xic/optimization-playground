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
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks
import os
import itertools

class MultipleGpuBigModelWrapper(abc.ABC):
    def start(self) -> None:
        self.rank = int(os.getenv("RANK", -1))
        self.world_size = int(os.getenv("WORLD_SIZE", 4))
        self.chunks = 4
        self._main()

        self.losses = []

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

    def run(self, module: torch.nn.Module, datloader, split_spec):
        X, y = next(iter(datloader))

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
        print(f"Pipeline stage {self.rank} {get_parameter_count(smod) / 10 ** 6}M params")

        stage = pipe.build_stage(
            self.rank,
            device=self.device,
        )

        optimizer = torch.optim.Adam(smod.parameters())

        # Attach to a schedule
        schedule = ScheduleGPipe(
            stage, 
            self.chunks,
            loss_fn=torch.nn.functional.cross_entropy
        )
        # tunr training back on so we can train this model ... 
        smod.train()
        for index, (X, y) in enumerate(datloader):
            X = X.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad(set_to_none=True)

            if self.rank == 0:            
                assert X.shape[0] == y.shape[0]
                schedule.step(
                    x=X,
                    target=y,
                    losses=self.losses,
                )
            else:
                out = schedule.step(
                    target=y,
                    losses=self.losses,
                )
                if index % 32 == 0:
                    print(sum([i.item() for i in self.losses]), (torch.sum(
                        torch.argmax(out, dim=1) == y
                    )) / y.shape[0] * 100)
            optimizer.step()

        dist.barrier()
        dist.destroy_process_group()
        
    def _main(self):
        if not "RANK" in os.environ:
            print(f"Not executed correctly. Instead run: ")
            print(f"torchrun --nproc-per-node 2 [file.py]")
            exit(0)

        backend = "nccl"
        dev_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{dev_id}")
        dist.init_process_group(
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device
        )
