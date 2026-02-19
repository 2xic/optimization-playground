"""
Loads in tensor from a hosted server
"""

import asyncio
import threading
import queue
import requests
import torch
from typing import Iterator, List, Optional
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
import ormsgpack
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FloatColumn:
    name: str

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class WebDataloader:
    def __init__(
        self,
        base_url,
        dataset_name,
        split="train",
        rank=0,
        world_size=1,
        columns: Optional[List[str]] = None,
        batch_size: int = 32,
        prefetch_factor: int = 16,
        max_workers: int = 8,
        timeout: int = 30,
    ):
        self.name = dataset_name
        self.base_url = base_url
        self.dataset_name = dataset_name
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.columns = columns
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.max_workers = max_workers
        self.timeout = timeout

        self.session = requests.Session()
        self._info = None

        # Iterator state
        self.epoch = 0
        self.batch_queue = queue.Queue(maxsize=prefetch_factor * 2)
        self.shutdown_event = threading.Event()
        self.async_thread = None
        self.batch_permutation = None
        self._failed_fetches = 0

        # Checkpointing state
        self._batches_consumed = 0
        self._resume_from = 0
        self._remaining_batches = None

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
    def total_rows(self):
        return self.info["num_rows"]

    @property
    def total_batches(self):
        return self.total_samples // (self.world_size * self.batch_size)

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
        if self._remaining_batches is not None:
            return self._remaining_batches
        return self.total_batches

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "batches_consumed": self._batches_consumed,
        }

    def load_state_dict(self, epoch, batches_consumed):
        self.epoch = epoch
        self._resume_from = batches_consumed
        self._batches_consumed = batches_consumed
        self._remaining_batches = self.total_batches - self._resume_from

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        if self.async_thread and self.async_thread.is_alive():
            self.async_thread.join(timeout=5.0)
            if self.async_thread.is_alive():
                logger.warning("Previous async thread did not finish in time")

        self.shutdown_event.clear()
        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break

        self._batches_consumed = self._resume_from
        self._failed_fetches = 0

        g = torch.Generator()
        g.manual_seed(42 + self.epoch)
        total_global_batches = self.total_batches * self.world_size
        self.batch_permutation = torch.randperm(
            total_global_batches, generator=g
        ).tolist()

        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()

        return self

    def __next__(self):
        try:
            while True:
                batch = self.batch_queue.get(timeout=60.0)
                if batch is None:
                    self.epoch += 1
                    raise StopIteration

                first_tensor = None
                for v in batch.values():
                    if torch.is_tensor(v):
                        first_tensor = v
                        break

                if first_tensor is not None and first_tensor.numel() > 0:
                    self._batches_consumed += 1
                    return batch

                self._failed_fetches += 1

        except queue.Empty:
            logger.warning("Timeout waiting for batch")
            raise StopIteration

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._fetch_all_batches())
        except Exception:
            logger.exception("Error in async fetch loop")
        finally:
            loop.close()

    async def _fetch_all_batches(self):
        connector = aiohttp.TCPConnector(
            limit=self.max_workers,
            limit_per_host=self.max_workers,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        retry_options = ExponentialRetry(attempts=5, start_timeout=2, max_timeout=30)
        async with RetryClient(
            connector=connector, timeout=timeout, retry_options=retry_options
        ) as session:
            start = self._resume_from
            self._resume_from = 0
            self._remaining_batches = None

            tasks = set()
            next_to_launch = start
            queued = 0
            total = self.total_batches

            while queued < total and not self.shutdown_event.is_set():
                while len(tasks) < self.prefetch_factor and next_to_launch < start + total:
                    tasks.add(asyncio.ensure_future(self._fetch_batch(session, next_to_launch)))
                    next_to_launch += 1

                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = task.result()
                    except Exception:
                        self._failed_fetches += 1
                        result = self._empty_batch()

                    if result is not None and not self.shutdown_event.is_set():
                        self.batch_queue.put(result)
                        queued += 1

            if not self.shutdown_event.is_set():
                self.batch_queue.put(None)

    async def _fetch_batch(self, session: aiohttp.ClientSession, batch_idx: int):
        if self.shutdown_event.is_set():
            return None

        permutation_idx = batch_idx * self.world_size + self.rank
        global_batch_idx = self.batch_permutation[permutation_idx]

        start_idx = global_batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_samples)

        columns = ",".join(list(map(str, self.column_names)))
        url = f"{self.base_url}/datasets/{self.dataset_name}/{self.split}/get?start={start_idx}&end={end_idx}&columns={columns}"

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()

            batch = ormsgpack.unpackb(content)
            result = {"dataset": self.name}

            for col in self.column_names:
                if isinstance(col, FloatColumn):
                    arr = np.array([item[col.name] for item in batch], dtype=np.float32)
                    col = col.name
                else:
                    arr = np.array([item[col] for item in batch], dtype=np.int64)
                result[col] = (
                    torch.from_numpy(arr).pin_memory()
                    if torch.cuda.is_available()
                    else torch.from_numpy(arr)
                )

            return result

        except Exception:
            return self._empty_batch()

    def _empty_batch(self):
        return {col: torch.tensor([]) for col in self.column_names}

    def cleanup(self):
        self.shutdown_event.set()
        if self.async_thread and self.async_thread.is_alive():
            self.async_thread.join(timeout=2.0)
        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break

    def __del__(self):
        self.cleanup()

    def convert_token_to_id(self, token):
        tokenized = self.tokenize([token], padding=False)
        assert len(tokenized) == 1
        assert len(tokenized[0]) == 1
        return tokenized[0][0]

    def tokenize(self, documents: List[str], padding=True):
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
            if padding:
                chunks = [
                    chunk + [self.padding_index] * (self.sequence_size - len(chunk))
                    for chunk in chunks
                ]
            results.append(torch.tensor(chunks, dtype=torch.long))

        return results

    def detokenize(self, token_ids: List[int]):
        assert isinstance(token_ids, list)
        token_ids = list(filter(lambda x: x != self.padding_index, token_ids))
        assert isinstance(token_ids, list)

        response = self.session.post(
            f"{self.base_url}/datasets/{self.dataset_name}/detokenize",
            json={"token_ids": token_ids},
        )

        if response.status_code != 200:
            raise Exception(response.text)

        return response.json()["text"]
