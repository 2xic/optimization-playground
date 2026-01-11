import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass, asdict
import json
import torch
import paramiko
import io
from dataclasses import dataclass, asdict
from datetime import date
import stat
from tqdm import tqdm
import hashlib
from functools import cache
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import date
import atexit
import time
from typing import Dict

load_dotenv()


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        key = (cls, args, tuple(sorted(kwargs.items())))
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]


class StorageBox(metaclass=SingletonMeta):
    def __init__(
        self, host: str, username: str, password: str, remote_dir: str = "checkpoints"
    ):
        self.remote_dir = remote_dir
        self.password = password

        self.cache_dir = Path("/tmp/sftp_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.transport = paramiko.Transport((host, 23))
        self.transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def _path_exists(self, path: str) -> bool:
        try:
            self.sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    def save_bytes(self, data: bytes, path: str):
        self.create_directory_recursive(str(Path(path).parent))
        with self.sftp.open(path, "wb") as f:
            f.write(data)

    def load_bytes(self, path: str, use_cache: bool = True) -> bytes:
        if use_cache:
            cache_path = self.cache_dir / self._cache_key(path)
            if cache_path.exists():
                return cache_path.read_bytes()

        with self.sftp.open(path, "rb") as f:
            file_size = f.stat().st_size
            chunks = []

            with tqdm(total=file_size, unit="B", unit_scale=True, desc=path) as pbar:
                while True:
                    chunk = f.read(32768)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    pbar.update(len(chunk))

        data = b"".join(chunks)

        if use_cache:
            cache_path.write_bytes(data)

        return data

    def delete(self, path: str):
        self.sftp.remove(path)

    def list(self, prefix=None) -> list[str]:
        directory = self.remote_dir
        if prefix is not None:
            directory = prefix
        return [os.path.join(directory, i) for i in self.sftp.listdir(directory)]

    def is_directory(self, path):
        file = self.sftp.stat(path)
        is_dir = stat.S_ISDIR(file.st_mode)
        return is_dir

    def close(self):
        self.sftp.close()
        self.transport.close()

    def create_directory(self, directory: str):
        if not self._path_exists(directory):
            self.sftp.mkdir(directory)

    def create_directory_recursive(self, directory: str):
        parts = directory.split("/")
        current = ""
        for part in parts:
            if not part:
                continue
            current = f"{current}/{part}" if current else part
            if not self._path_exists(current):
                self.sftp.mkdir(current)

    @cache
    def walk(self, base=None, max_age_days=None, min_age_days=None):
        queue = [base]
        now = time.time()
        max_age_cutoff = (
            now - (max_age_days * 86400) if max_age_days is not None else None
        )
        min_age_cutoff = (
            now - (min_age_days * 86400) if min_age_days is not None else None
        )

        def skip_file_or_folder(item):
            mtime = self.sftp.stat(item).st_mtime
            if max_age_cutoff is not None and mtime < max_age_cutoff:
                return True
            if min_age_cutoff is not None and mtime > min_age_cutoff:
                return True
            return False

        while queue:
            current = queue.pop()
            for item in self.list(current):
                if skip_file_or_folder(item):
                    continue
                if self.is_directory(item):
                    queue.append(item)
                    continue
                yield item

    def _cache_key(self, path: str) -> str:
        stat = self.sftp.stat(path)
        key_str = f"{path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(key_str.encode()).hexdigest()


@dataclass
class Stats:
    accuracy_pct: float
    loss_average: float
    runtime_seconds: int
    steps: int
    dataset: str
    metadata: Dict[str, str]

    def to_json(self) -> dict:
        return asdict(self)


# Uploads files in background to not block the training loop.
class StorageBoxCheckpoint(StorageBox):
    def __init__(self, host, username, password, run_id, max_workers=2):
        today = date.today().isoformat()
        self.base_name = f"checkpoints/{today}/{run_id}"
        super().__init__(host, username, password, self.base_name)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        atexit.register(self._shutdown)

    def _shutdown(self):
        self._executor.shutdown(wait=True)

    def checkpoint(self, model, optimizer, config, stats: Stats) -> Future:
        full_path = os.path.join(self.base_name, f"step_{stats.steps}")
        files = {
            "model.pt": self._serialize_torch(model),
            "optimizer.pt": self._serialize_torch(optimizer),
            "stats.json": self._serialize_json(stats),
            "config.json": self._serialize_json(config),
        }

        return self._executor.submit(self._upload, files, full_path)

    def _upload(self, files, full_path):
        for name, data in files.items():
            try:
                self.save_bytes(data, os.path.join(full_path, name))
            except Exception as e:
                print(e)

    def _serialize_json(self, data) -> bytes:
        return json.dumps(data.to_json(), indent=2).encode("utf-8")

    def _serialize_torch(self, torch_state) -> bytes:
        buffer = io.BytesIO()
        torch.save(torch_state.state_dict(), buffer)
        return buffer.getvalue()


if __name__ == "__main__":
    checkpointer = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    # checkpointer.list()
    #    with open("local_checkpoint2.pth", "w") as file:
    #        file.write("tset")
    #    checkpointer.save("local_checkpoint2.pth")
    #   checkpointer.delete("local_checkpoint.pth")
    #    print("Available checkpoints:", checkpointer.list())
    print(
        checkpointer.load_bytes(
            "20251212_204548/step_0/stats.json"  # , "/tmp/stats.json"
        )
    )
    checkpointer.close()
