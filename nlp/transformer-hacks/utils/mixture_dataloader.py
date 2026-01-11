import random
from typing import List, Iterator
from .web_dataloader import WebDataloader, ThreadedDataLoader


class WebDataloaderMixture:
    def __init__(
        self,
        dataloaders: List[WebDataloader],
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataloaders = dataloaders
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.name = "+".join(dl.name for dl in dataloaders)

        for dl in self.dataloaders:
            dl.rank = rank
            dl.world_size = world_size

        self.vocab_size = dataloaders[0].vocab_size
        self.padding_index = dataloaders[0].padding_index
        self.sequence_size = dataloaders[0].sequence_size

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def iter(self, batch_size: int = 4, workers: int = 16):
        workers_per_loader = max(1, workers // len(self.dataloaders))
        return MixtureIterator(
            loaders=[
                dl.iter(batch_size=batch_size, workers=workers_per_loader)
                for dl in self.dataloaders
            ],
            seed=self.seed,
            name=self.name,
        )


class MixtureIterator:
    def __init__(self, loaders: List[ThreadedDataLoader], seed: int, name: str):
        self.loaders = loaders
        self.seed = seed
        self.epoch = 0
        self.lengths = [len(loader) for loader in loaders]
        self.total = sum(self.lengths)
        self.name = name

    def set_batch_size(self, batch_size):
        for i in self.loaders:
            i.dataset.batch_size = batch_size

    def set_epoch(self, epoch):
        for i in self.loaders:
            i.iter.set_epoch(epoch)

    def __len__(self):
        return self.total

    def __iter__(self) -> Iterator:
        iters = [iter(loader) for loader in self.loaders]
        active = [True] * len(self.loaders)

        rng = random.Random(self.seed + self.epoch)
        loader_idx = rng.randrange(len(self.loaders))

        while any(active):
            if active[loader_idx]:
                try:
                    yield next(iters[loader_idx])
                except StopIteration:
                    active[loader_idx] = False
            loader_idx = (loader_idx + 1) % len(self.loaders)

    def __del__(self):
        for loader in self.loaders:
            if hasattr(loader, "session"):
                loader.session.close()
