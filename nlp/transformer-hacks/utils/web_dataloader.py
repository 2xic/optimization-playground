"""
Loads in tensor from a hosted server
"""

import threading
import queue
import requests
import torch
import time
from typing import Iterator, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import msgpack
import numpy as np


class WebDataloader:
    def __init__(
        self,
        base_url,
        dataset_name,
        split="train",
        batch_size=32,
        rank=0,
        world_size=1,
        columns: Optional[List[str]] = None,
    ):
        self.name = dataset_name
        self.base_url = base_url
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.columns = columns

        self.session = requests.Session()
        self._info = None

    @property
    def info(self):
        if self._info is None:
            print(f"{self.base_url}/datasets/{self.dataset_name}/{self.split}/info")
            response = self.session.get(
                f"{self.base_url}/datasets/{self.dataset_name}/{self.split}/info"
            )
            self._info = response.json()
        return self._info

    @property
    def column_names(self):
        if self.columns is not None:
            return self.columns
        return self.info.get("columns", ["x_tokens", "y_tokens"])

    @property
    def total_samples(self):
        return self.info["num_rows"]

    @property
    def num_batches(self):
        return self.total_samples // (self.world_size * self.batch_size)

    @property
    def total_rows(self):
        return self.info["num_rows"]

    @property
    def vocab_size(self):
        return self.info["training_metadata"]["vocab_size"]

    @property
    def padding_index(self):
        return self.info["training_metadata"]["padding_index"]

    @property
    def sequence_size(self):
        return self.info["training_metadata"]["sequence_size"]

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

    def tokenize(self, documents: List[str]):
        assert isinstance(documents, list)

        response = self.session.post(
            f"{self.base_url}/datasets/{self.dataset_name}/tokenize",
            json={"documents": documents},
        )

        if response.status_code != 200:
            raise Exception(response.text)

        tokenized_docs = response.json()["tokenized_documents"]

        results = []
        for doc in tokenized_docs:
            chunks = [
                doc[start : start + self.sequence_size]
                for start in range(0, len(doc), self.sequence_size)
            ]
            padded = [
                chunk + [self.padding_index] * (self.sequence_size - len(chunk))
                for chunk in chunks
            ]
            results.append(torch.tensor(padded, dtype=torch.long))

        return results

    def detokenize(self, token_ids: List[str]):
        assert isinstance(token_ids, list)

        response = self.session.post(
            f"{self.base_url}/datasets/{self.dataset_name}/detokenize",
            json={"token_ids": token_ids},
        )

        if response.status_code != 200:
            raise Exception(response.text)

        return response.json()["text"]


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

    def set_epoch(self, epoch):
        self.iter.set_epoch(epoch)

    def set_batch_size(self, batch_size):
        self.iter.set_batch_size(batch_size)
        self.dataset.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator:
        self.iter.set_epoch(self.epoch)
        self.epoch += 1
        return iter(self.iter)

    def __del__(self):
        if hasattr(self, "iter"):
            self.iter.cleanup()

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
        self.total_batches = total_batches
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

    def set_batch_size(self, batch_size):
        self.dataset.batch_size = batch_size

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

        global_batch_idx = self.batch_permutation[
            batch_idx * self.world_size + self.rank
        ]
        start_idx = global_batch_idx * self.dataset.batch_size
        end_idx = min(start_idx + self.dataset.batch_size, self.dataset.total_samples)

        columns = ",".join(self.dataset.column_names)
        url = f"{self.dataset.base_url}/datasets/{self.dataset.dataset_name}/{self.dataset.split}/get?start={start_idx}&end={end_idx}&columns={columns}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            batch = msgpack.unpackb(response.content, raw=False)
            columns = self.dataset.column_names

            # Build dict of tensors for each column
            result = {}
            for col in columns:
                arr = np.array([item[col] for item in batch], dtype=np.int64)
                result[col] = (
                    torch.from_numpy(arr).pin_memory()
                    if torch.cuda.is_available()
                    else torch.from_numpy(arr)
                )

            return batch_idx, result

        except Exception as e:
            if not self.shutdown:
                print(f"Error fetching batch {batch_idx}: {e}")
            # Return empty tensors for each column
            columns = self.dataset.column_names
            return batch_idx, {col: torch.tensor([]) for col in columns}

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
            while True:
                batch = self.batch_queue.get(timeout=60.0)

                first_col = next(iter(batch.values()))
                if first_col.numel() == 0:
                    if self.next_expected_idx >= self.total_batches:
                        self.epoch_finished = True
                        raise StopIteration
                    continue

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

        if self.manager_thread.is_alive():
            self.manager_thread.join(timeout=2.0)

        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break

        self.executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        self.cleanup()
