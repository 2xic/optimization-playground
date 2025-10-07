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
import torch
import torch
import msgpack
import requests
import time
import torch.distributed as dist
import numpy as np


class WebDataloader:
    def __init__(self, base_url, dataset_name, split="train", batch_size=32):
        self.name = dataset_name
        self.base_url = base_url
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size

        self.session = requests.Session()  # Reuse connections

        # Get dataset info to know total size
        response = self.session.get(
            f"{self.base_url}/datasets/{self.dataset_name}/{self.split}/info"
        )
        self.info = response.json()
        self.total_samples = self.info["num_rows"]

        # Calculate number of batches
        self.num_batches = (self.total_samples + self.batch_size - 1) // self.batch_size
        self.total_rows = self.info["num_rows"]
        self.vocab_size = self.info["training_metadata"]["vocab_size"]
        self.padding_index = self.info["training_metadata"]["padding_index"]
        self.sequence_size = self.info["training_metadata"]["sequence_size"]

    def __len__(self):
        return self.num_batches

    def iter(self, batch_size=4, workers=8):
        self.batch_size = batch_size
        return ThreadedDataLoader(dataset=self, prefetch_factor=16, max_workers=workers)


class ThreadedDataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        prefetch_factor: int = 4,
        max_workers: int = 8,
        timeout: int = 30,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.max_workers = max_workers
        self.timeout = timeout
        self.total_batches = len(dataset)
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
        )

    def __len__(self):
        return self.total_batches

    def __iter__(self) -> Iterator:
        return iter(self.iter)

    def __del__(self):
        if hasattr(self, "session"):
            self.session.close()


class ThreadedIterator:
    def __init__(
        self, dataset, session, total_batches, prefetch_factor, max_workers, timeout
    ):
        self.dataset = dataset
        self.session = session
        self.total_batches = total_batches
        self.prefetch_factor = prefetch_factor
        self.max_workers = max_workers
        self.timeout = timeout

        # Persistent components - don't recreate between epochs
        self.batch_queue = queue.Queue(maxsize=prefetch_factor)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.shutdown = False

        # Per-epoch state - reset on each __iter__
        self.current_batch_idx = 0
        self.futures = {}
        self.epoch_finished = False

        # Start the persistent manager thread
        self.manager_thread = threading.Thread(
            target=self._manage_completed_requests, daemon=True
        )
        self.manager_thread.start()

    def _start_epoch(self):
        """Reset state and start fetching for a new epoch"""
        # Clear any remaining items from queue
        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break

        # Reset epoch state
        self.current_batch_idx = 0
        self.epoch_finished = False
        self.next_expected_idx = 0
        self.completed_batches = {}

        # Start prefetching first few batches
        batches_to_prefetch = min(self.prefetch_factor, self.total_batches)
        for i in range(batches_to_prefetch):
            if not self.shutdown:
                future = self.executor.submit(self._fetch_batch, i)
                self.futures[future] = i

        self.current_batch_idx = batches_to_prefetch

    def _fetch_batch(self, batch_idx: int):
        if self.shutdown:
            return None

        start_idx = batch_idx * self.dataset.batch_size
        end_idx = min(start_idx + self.dataset.batch_size, self.dataset.total_samples)
        indices = list(range(start_idx, end_idx))
        indices_str = ",".join(map(str, indices))

        url = f"{self.dataset.base_url}/datasets/{self.dataset.dataset_name}/{self.dataset.split}/get?indices={indices_str}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            batch = msgpack.unpackb(response.content, raw=False)

            # Stack as numpy first (faster), then convert to torch
            x_array = np.array([item["x_tokens"] for item in batch], dtype=np.int64)
            y_array = np.array([item["y_tokens"] for item in batch], dtype=np.int64)

            # Convert to pinned torch tensors
            x_tokens = torch.from_numpy(x_array).pin_memory()
            y_tokens = torch.from_numpy(y_array).pin_memory()
            return batch_idx, (x_tokens, y_tokens)

        except Exception as e:
            if not self.shutdown:
                print(f"Error fetching batch {batch_idx}: {e}")
            return batch_idx, (torch.tensor([]), torch.tensor([]))

    def _manage_completed_requests(self):
        """Background thread that manages completed HTTP requests - runs across epochs"""
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

                            # Submit next batch request if available
                            if (
                                self.current_batch_idx < self.total_batches
                                and not self.shutdown
                            ):
                                next_future = self.executor.submit(
                                    self._fetch_batch, self.current_batch_idx
                                )
                                self.futures[next_future] = self.current_batch_idx
                                self.current_batch_idx += 1

                            # Put completed batches in order into the queue
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
                                    # Queue is full, put it back and try again later
                                    self.completed_batches[self.next_expected_idx] = (
                                        batch_data
                                    )
                                    break
                        break
                else:
                    time.sleep(0.1)

            except Exception as e:
                # if not self.shutdown:
                #    print(f"Manager thread error: {e}")
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
                raise TimeoutError("Timeout waiting for batch")

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
