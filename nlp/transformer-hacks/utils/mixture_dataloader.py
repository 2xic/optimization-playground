import random
from collections.abc import Iterator
from .web_dataloader import WebDataloader


class WebDataloaderMixture:
    def __init__(
        self,
        dataloaders: list[WebDataloader],
        seed: int = 42,
    ):
        self.dataloaders = dataloaders
        self.seed = seed
        self.name = "+".join(dl.name for dl in dataloaders)

        self.vocab_size = dataloaders[0].vocab_size
        self.padding_index = dataloaders[0].padding_index
        self.sequence_size = dataloaders[0].sequence_size

        # Iterator state
        self.epoch = 0
        self._active = None
        self._iters = None
        self._rng = None

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    @property
    def _batches_consumed(self):
        return sum(dl._batches_consumed for dl in self.dataloaders)

    @property
    def total_batches(self):
        return sum(dl.total_batches for dl in self.dataloaders)

    @property
    def _failed_fetches(self):
        return sum(dl._failed_fetches for dl in self.dataloaders)

    class _AggQueue:
        def __init__(self, dls):
            self._dls = dls
        def qsize(self):
            return sum(dl.batch_queue.qsize() for dl in self._dls)

    @property
    def batch_queue(self):
        return WebDataloaderMixture._AggQueue(self.dataloaders)

    def set_batch_size(self, batch_size):
        for dl in self.dataloaders:
            dl.set_batch_size(batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch
        for dl in self.dataloaders:
            dl.set_epoch(epoch)

    def load_state_dict(self, epoch, batches_consumed):
        self.epoch = epoch
        for dl in self.dataloaders:
            dl.load_state_dict(epoch=epoch, batches_consumed=batches_consumed)

    def __iter__(self) -> Iterator:
        self._iters = [iter(dl) for dl in self.dataloaders]
        self._active = [True] * len(self.dataloaders)
        self._rng = random.Random(self.seed + self.epoch)

        return self

    def __next__(self):
        if not any(self._active):
            raise StopIteration

        start_idx = self._rng.randrange(len(self.dataloaders))
        for offset in range(len(self.dataloaders)):
            loader_idx = (start_idx + offset) % len(self.dataloaders)
            if self._active[loader_idx]:
                try:
                    return next(self._iters[loader_idx])
                except StopIteration:
                    self._active[loader_idx] = False

        raise StopIteration

    def cleanup(self):
        for dl in self.dataloaders:
            dl.cleanup()

    def __del__(self):
        self.cleanup()
