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
import os
import atexit
from tqdm import tqdm

class MultipleGpuBigModelWrapper(abc.ABC):
    def __init__(self, loss_function) -> None:
        super().__init__()
        self.loss_function = loss_function

    def start(self) -> None:
        self.rank = int(os.getenv("RANK", -1))
        self.world_size = int(os.getenv("WORLD_SIZE", 4))
        self.chunks = self.world_size
        self._main()

        self.losses = []
        self.epoch = 0
        self.train = True
        self.stage = None
        self.batch_X = None
        self.batch_y = None

        atexit.register(self._destroy)

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

        input_X = X[:X.shape[0] // self.chunks].reshape((X.shape[0] // self.chunks, ) + tuple(X.shape[1:]))
        input_Y = y[:y.shape[0] // self.chunks].reshape((y.shape[0] // self.chunks, ) + tuple(y.shape[1:]))
        assert input_X.shape[0] == X.shape[0] // self.chunks
        assert input_Y.shape[0] == y.shape[0] // self.chunks

        self.pipe: Pipe = pipeline(
            module=module,
            mb_args=(),
            mb_kwargs={
                "x": input_X,
            },
            split_spec=split_spec
        )
        smod = self.pipe.get_stage_module(self.rank)

        self.stage = self.pipe.build_stage(
            self.rank,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(smod.parameters())#, lr=1e-4)

        # Attach to a schedule
        self.schedule = ScheduleGPipe(
            self.stage, 
            self.chunks,
            loss_fn=(self.loss_function if self.train else None)
        )

        # turn training back on so we can train this model ... 
        smod.train()

    def run_epoch(self, dataloader, epochs, view_function=lambda x: x):
        self.schedule._has_backward = True
        for _ in range(epochs):
            dataloader_iterator = tqdm(dataloader) if self.rank == 0 else dataloader
            for _, (X, y) in enumerate(dataloader_iterator):
                self.optimizer.zero_grad(set_to_none=True)
                X = X.to(self.device)
                y = y.to(self.device)
                # use drop_last=True and shuffle to not trigger
                assert X.shape[0] == y.shape[0]

                out = self.forward(X, y, view_function)
                if out is not None:
                    self.batch_done(self.losses, X, view_function(y), out)
                self.optimizer.step()
            # Store the last batch so we can use it for predictions
            self.batch_X = X
            self.batch_y = y
            # Need to do the first epoch to make sure we are at a good stage
            self.epoch_done(self.epoch, self.stage.is_last)
            # want to update all as there could be conditionals in epoch_done
            self.epoch += 1

    @property
    def module(self):
        return self.pipe.get_stage_module(self.rank)

    def load(self):
        print("Loading in module state")
        if os.path.isfile(self.stage_name):
            state = torch.load(self.stage_name, weights_only=True)
            smod = self.module
            smod.load_state_dict(state)

    def save(self):
        smod = self.module
        torch.save(smod.state_dict(), self.stage_name)

    @property
    def stage_name(self):
        return f"model_stage_{self.rank}_state_dict.pth"

    # destroyed by an at exit hook
    def _destroy(self):
        dist.destroy_process_group()

    def forward(self, X, y=None, view_function=lambda x: x):
        train = y != None
        # disable the loss calculation if not used
        self.schedule._has_backward = train
        if self.rank == 0:
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
                assert y is None or out.shape[0] == view_function(y).shape[0], f"Mismatch with view function {out.shape} / {view_function(y).shape} and X ({X.shape})"
                return out
        return None

    def _main(self):
        if not "RANK" in os.environ:
            print(f"Not executed correctly. Instead run: ")
            print(f"torchrun --nproc-per-node {torch.cuda.device_count()} [file.py]")
            print(f"torchrun --nproc-per-node {torch.cuda.device_count()} -m module.file")
            exit(0)

        dev_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{dev_id}")

        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device,
        )

    def batch_done(self, losses, X: torch.Tensor, y: torch.Tensor, y_predicted: torch.Tensor):
        loss = sum([i.item() for i in losses])
        print(f"Loss {loss}")

    def epoch_done(self, epoch, is_last_stage):
        pass

    def log(self, *args):
        if self.rank == 0:
            print(*args)
