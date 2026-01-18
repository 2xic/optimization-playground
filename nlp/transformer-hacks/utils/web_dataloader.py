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


class ThreadedDataLoader:
    def __init__(
        self,
        dataset: WebDataloader,
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

        self.batch_queue = queue.Queue(maxsize=prefetch_factor * 2)
        self.shutdown_event = threading.Event()

        self.async_thread = None
        self.batch_permutation = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.dataset.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

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

        g = torch.Generator()
        g.manual_seed(42 + self.epoch)
        self.batch_permutation = torch.randperm(
            self.total_batches, generator=g
        ).tolist()

        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()

        self.epoch += 1
        return self

    def __next__(self):
        try:
            while True:
                batch = self.batch_queue.get(timeout=60.0)
                if batch is None:
                    raise StopIteration

                first_tensor = None
                for v in batch.values():
                    if torch.is_tensor(v):
                        first_tensor = v
                        break

                if first_tensor is not None and first_tensor.numel() > 0:
                    return batch

                logger.warning("Skipping empty batch from failed fetch")

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

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            batch_idx = 0

            while batch_idx < self.total_batches and not self.shutdown_event.is_set():
                chunk_size = min(self.prefetch_factor, self.total_batches - batch_idx)
                tasks = [
                    self._fetch_batch(session, batch_idx + i) for i in range(chunk_size)
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                fetched = {}
                for i, result in enumerate(results):
                    idx = batch_idx + i
                    if isinstance(result, Exception):
                        logger.exception(
                            f"Fetch failed for batch {idx}", exc_info=result
                        )
                        fetched[idx] = self._empty_batch()
                    elif result is not None:
                        fetched[idx] = result

                for i in range(chunk_size):
                    idx = batch_idx + i
                    if idx in fetched and not self.shutdown_event.is_set():
                        self.batch_queue.put(fetched[idx])

                batch_idx += chunk_size

            if not self.shutdown_event.is_set():
                self.batch_queue.put(None)

    async def _fetch_batch(self, session: aiohttp.ClientSession, batch_idx: int):
        if self.shutdown_event.is_set():
            return None

        global_batch_idx = self.batch_permutation[
            batch_idx * self.world_size + self.rank
        ]
        start_idx = global_batch_idx * self.dataset.batch_size
        end_idx = min(start_idx + self.dataset.batch_size, self.dataset.total_samples)

        columns = ",".join(list(map(str, self.dataset.column_names)))
        url = f"{self.dataset.base_url}/datasets/{self.dataset.dataset_name}/{self.dataset.split}/get?start={start_idx}&end={end_idx}&columns={columns}"

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()

            batch = ormsgpack.unpackb(content)
            result = {"dataset": self.dataset.name}

            for col in self.dataset.column_names:
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
            logger.exception(f"Error fetching batch {batch_idx} from {url}")
            return self._empty_batch()

    def _empty_batch(self):
        return {col: torch.tensor([]) for col in self.dataset.column_names}

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
