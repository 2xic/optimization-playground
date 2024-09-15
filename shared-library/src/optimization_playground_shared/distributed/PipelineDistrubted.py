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

# x = torch.LongTensor([1, 2, 4, 5])

class MultipleGpuBigModelWrapper(abc.ABC):
    def start(self, is_debug_mode=False) -> None:
        self.rank = int(os.getenv("RANK", -1))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.chunks = 1
        self._main(gpu_id=0, world_size=None, is_debug_mode=is_debug_mode)

    def run(self, module, inputs):
        X, y = inputs

        X = X.to(self.device)

        module.to(self.device)
        module.eval()

       # print((next(module.parameters()).device, X.device))

        pipe: Pipe = pipeline(
            module=module,
            mb_args=(),
            mb_kwargs={
                "x": X[0].reshape((1, 1, 28, 28)),
            },
            split_spec={
                "conv1": SplitPoint.BEGINNING,
            }
        )
        smod = pipe.get_stage_module(self.rank)
        print(f"Pipeline stage {self.rank} {get_parameter_count(smod) / 10 ** 6}M params")

        stage = pipe.build_stage(
            self.rank,
            device=self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(
            stage, 
            self.chunks,
        )
        
        if self.rank == 0:            
            _, kwargs_split = split_args_kwargs_into_chunks(
                args=(),
                kwargs={
                    "x": X# [0].reshape((1, 1, 28, 28)),
                },
                chunks=self.chunks,
                #args_chunk_spec=args_chunk_spec,
                #kwargs_chunk_spec=kwargs_chunk_spec
            )
            schedule.step(
                x=X, #kwargs_split
            )
        else:
            out = schedule.step()
            print(out)
            pass 

        dist.barrier()
        dist.destroy_process_group()
        
    def _main(self, gpu_id, world_size, is_debug_mode):
        backend = "nccl"
        print(gpu_id)
        dev_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{dev_id}")
        dist.init_process_group(
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device
        )
