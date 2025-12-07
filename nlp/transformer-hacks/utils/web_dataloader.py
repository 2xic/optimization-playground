"""
Loads in tensor from a hosted server
"""

import threading
import queue
import requests
import torch
import time
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import msgpack
import numpy as np


class WebDataloader:
    def __init__(
        self, base_url, dataset_name, split="train", batch_size=32, rank=0, world_size=1
    ):
        self.name = dataset_name
        self.base_url = base_url
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        self.session = requests.Session()

        response = self.session.get(
            f"{self.base_url}/datasets/{self.dataset_name}/{self.split}/info"
        )
        self.info = response.json()
        self.total_samples = self.info["num_rows"]

        self.num_batches = (self.total_samples + self.batch_size - 1) // self.batch_size
        self.total_rows = self.info["num_rows"]
        self.vocab_size = self.info["training_metadata"]["vocab_size"]
        self.padding_index = self.info["training_metadata"]["padding_index"]
        self.sequence_size = self.info["training_metadata"]["sequence_size"]

    def __len__(self):
        return self.num_batches

    def iter(self, batch_size=4, workers=16):
        self.batch_size = batch_size
        return ThreadedDataLoader(
            dataset=self,
            prefetch_factor=16,
            max_workers=workers,
            rank=self.rank,
            world_size=self.world_size,
        )


class ThreadedDataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        prefetch_factor: int = 4,
        max_workers: int = 8,
        timeout: int = 30,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.name = dataset.name
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.max_workers = max_workers
        self.timeout = timeout
        self.total_batches = len(dataset)
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers, pool_maxsize=max_workers * 2, max_retries=3
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.iter = ThreadedIterator(
            dataset=self.dataset,
            session=self.session,
            total_batches=self.total_batches,
            prefetch_factor=self.prefetch_factor,
            max_workers=self.max_workers,
            timeout=self.timeout,
            rank=self.rank,
            world_size=self.world_size,
        )

    def __len__(self):
        return self.total_batches // self.world_size

    def __iter__(self) -> Iterator:
        self.iter.set_epoch(self.epoch)
        self.epoch += 1
        return iter(self.iter)

    def __del__(self):
        if hasattr(self, "session"):
            self.session.close()


class ThreadedIterator:
    def __init__(
        self,
        dataset,
        session,
        total_batches,
        prefetch_factor,
        max_workers,
        timeout,
        rank=0,
        world_size=1,
    ):
        self.dataset = dataset
        self.session = session
        self.total_batches = total_batches // world_size
        self.total_global_batches = total_batches
        self.prefetch_factor = prefetch_factor
        self.max_workers = max_workers
        self.timeout = timeout
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.batch_permutation = None

        self.batch_queue = queue.Queue(maxsize=prefetch_factor)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.shutdown = False

        self.current_batch_idx = 0
        self.futures = {}
        self.epoch_finished = False

        self.manager_thread = threading.Thread(
            target=self._manage_completed_requests, daemon=True
        )
        self.manager_thread.start()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _start_epoch(self):
        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break

        self.current_batch_idx = 0
        self.epoch_finished = False
        self.next_expected_idx = 0
        self.completed_batches = {}

        # Shuffle batch indices — same seed on all ranks so they don't overlap
        g = torch.Generator()
        g.manual_seed(42 + self.epoch)
        self.batch_permutation = torch.randperm(
            self.total_global_batches, generator=g
        ).tolist()

        batches_to_prefetch = min(self.prefetch_factor, self.total_batches)
        for i in range(batches_to_prefetch):
            if not self.shutdown:
                future = self.executor.submit(self._fetch_batch, i)
                self.futures[future] = i

        self.current_batch_idx = batches_to_prefetch

    def _fetch_batch(self, batch_idx: int):
        if self.shutdown:
            return None

        # Get shuffled global batch index for this rank
        global_batch_idx = self.batch_permutation[
            batch_idx * self.world_size + self.rank
        ]
        start_idx = global_batch_idx * self.dataset.batch_size
        end_idx = min(start_idx + self.dataset.batch_size, self.dataset.total_samples)

        url = f"{self.dataset.base_url}/datasets/{self.dataset.dataset_name}/{self.dataset.split}/get?start={start_idx}&end={end_idx}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            batch = msgpack.unpackb(response.content, raw=False)

            x_array = np.array([item["x_tokens"] for item in batch], dtype=np.int64)
            y_array = np.array([item["y_tokens"] for item in batch], dtype=np.int64)

            x_tokens = torch.from_numpy(x_array).pin_memory()
            y_tokens = torch.from_numpy(y_array).pin_memory()
            return batch_idx, (x_tokens, y_tokens)

        except Exception as e:
            if not self.shutdown:
                print(f"Error fetching batch {batch_idx}: {e}")
            return batch_idx, (torch.tensor([]), torch.tensor([]))

    def _manage_completed_requests(self):
        while not self.shutdown:
            try:
                if self.futures:
                    for future in as_completed(self.futures, timeout=1.0):
                        if self.shutdown:
                            break

                        _ = self.futures.pop(future)
                        result = future.result()

                        if result and not self.epoch_finished:
                            completed_idx, batch_data = result
                            self.completed_batches[completed_idx] = batch_data

                            if (
                                self.current_batch_idx < self.total_batches
                                and not self.shutdown
                            ):
                                next_future = self.executor.submit(
                                    self._fetch_batch, self.current_batch_idx
                                )
                                self.futures[next_future] = self.current_batch_idx
                                self.current_batch_idx += 1

                            while (
                                self.next_expected_idx in self.completed_batches
                                and not self.shutdown
                                and not self.epoch_finished
                            ):
                                batch_data = self.completed_batches.pop(
                                    self.next_expected_idx
                                )
                                try:
                                    self.batch_queue.put(batch_data, timeout=1.0)
                                    self.next_expected_idx += 1
                                except queue.Full:
                                    self.completed_batches[self.next_expected_idx] = (
                                        batch_data
                                    )
                                    break
                        break
                else:
                    time.sleep(0.1)

            except Exception as e:
                time.sleep(0.1)

    def __iter__(self):
        self._start_epoch()
        return self

    def __next__(self):
        try:
            batch = self.batch_queue.get(timeout=60.0)

            if self.next_expected_idx >= self.total_batches:
                self.epoch_finished = True

            return batch

        except queue.Empty:
            if self.next_expected_idx >= self.total_batches:
                self.epoch_finished = True
                raise StopIteration
            else:
                print("Timeout waiting for batch")
                self.epoch_finished = True
                raise StopIteration

    def cleanup(self):
        self.shutdown = True
        self.epoch_finished = True

        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break

        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def __del__(self):
        self.cleanup()
