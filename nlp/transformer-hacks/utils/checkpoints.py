import warnings
from cryptography.utils import CryptographyDeprecationWarning
from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass, asdict, field
import json
import torch
import paramiko
import io
from datetime import date
import stat
from tqdm import tqdm
import hashlib
from functools import cache
from concurrent.futures import ThreadPoolExecutor, Future
import atexit
import time
from typing import Dict
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import wait, FIRST_COMPLETED
import gzip
from paramiko.common import MAX_WINDOW_SIZE
import subprocess
import tempfile
import shutil

assert shutil.which("rsync"), "rsync not installed (apt install rsync)"
assert shutil.which("sshpass"), "sshpass not installed (apt install sshpass)"

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
load_dotenv()

INDEX_PATH = "checkpoints/index.ndjson"

Path("/tmp/sftp_cache").mkdir(parents=True, exist_ok=True)


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
        self.username = username
        self.host = host

        self.cache_dir = Path("/tmp/sftp_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.transport = paramiko.Transport((host, 23))
        self.transport.default_window_size = MAX_WINDOW_SIZE  # ~2GB
        self.transport.default_max_packet_size = 2**15  # 32KB packets

        self.transport.packetizer.REKEY_BYTES = 2**40
        self.transport.packetizer.REKEY_PACKETS = 2**40

        self.transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def _path_exists(self, path: str) -> bool:
        try:
            self.sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    # def save_bytes(self, data: bytes, path: str):
    #    #self.create_directory_recursive(str(Path(path).parent))
    #    #with self.sftp.open(path, "wb") as f:
    #    #    #chunk_size = 1024 * 1024  # 1 MiB chunks
    #    #    #for i in range(0, len(data), chunk_size):
    #    #    #    #f.write(data[i : i + chunk_size])
    #    #        f.flush()

    def save_bytes(self, data: bytes, path: str):
        self.create_directory_recursive(str(Path(path).parent))

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            env = os.environ.copy()
            env["SSHPASS"] = self.password
            # paramiko is so slow ...
            subprocess.run(
                [
                    "sshpass",
                    "-e",
                    "rsync",
                    "-e",
                    "ssh -p 23 -T -o Compression=no -o StrictHostKeyChecking=no",
                    tmp_path,
                    f"{self.username}@{self.host}:{path}",
                ],
                env=env,
                check=True,
            )
        finally:
            os.unlink(tmp_path)

    def load_bytes(self, path: str, use_cache: bool = True) -> bytes:
        if use_cache:
            cache_path = self.cache_dir / self._cache_key(path)
            if cache_path.exists():
                data = cache_path.read_bytes()
                if data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)
                return data

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            env = os.environ.copy()
            env["SSHPASS"] = self.password

            subprocess.run(
                [
                    "sshpass",
                    "-e",
                    "rsync",
                    "-e",
                    "ssh -p 23 -T -o Compression=no -o StrictHostKeyChecking=no",
                    "--progress",
                    f"{self.username}@{self.host}:{path}",
                    tmp_path,
                ],
                env=env,
                check=True,
            )

            data = Path(tmp_path).read_bytes()
        finally:
            os.unlink(tmp_path)

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)

        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)

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

    def append_index_entry(self, run_id: str, step: int, path: str, stats: dict = None):
        entry = {
            "run_id": run_id,
            "step": step,
            "path": path,
            "stats": stats,
            "indexed_at": datetime.now().isoformat(),
        }
        line = json.dumps(entry) + "\n"
        # TODO: ?
        # self.save_bytes(line.encode("utf-8"), INDEX_PATH)
        with self.sftp.open(INDEX_PATH, "a") as f:
            f.write(line.encode("utf-8"))

    def iterate_index(self):
        with self.sftp.open(INDEX_PATH, "r") as f:
            return [json.loads(i) for i in f.readlines()]


@dataclass
class TrainingHistory:
    losses: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)
    step_losses: list = field(default_factory=list, repr=False)
    step_accuracies: list = field(default_factory=list, repr=False)
    epoch_at_step: list = field(default_factory=list, repr=False)

    def record_epoch(self, loss, accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.epoch_at_step.append(len(self.step_losses))

    def record_step(self, loss, accuracy):
        self.step_losses.append(loss)
        self.step_accuracies.append(accuracy)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(
            losses=d.get("losses", []),
            accuracies=d.get("accuracies", []),
            step_losses=d.get("step_losses", []),
            step_accuracies=d.get("step_accuracies", []),
            epoch_at_step=d.get("epoch_at_step", []),
        )


class TrainingMetadata(dict):
    @property
    def plots(self) -> TrainingHistory:
        if "plots" not in self:
            self["plots"] = TrainingHistory()
        return self["plots"]

    @plots.setter
    def plots(self, value):
        self["plots"] = value

    @property
    def batches_consumed(self) -> int:
        return self.get("batches_consumed", 0)

    @batches_consumed.setter
    def batches_consumed(self, value):
        self["batches_consumed"] = value

    @property
    def epoch(self) -> int:
        return self.get("epoch", 0)

    @epoch.setter
    def epoch(self, value):
        self["epoch"] = value

    @property
    def sum_accuracy(self) -> float:
        return self.get("sum_accuracy", 0.0)

    @sum_accuracy.setter
    def sum_accuracy(self, value):
        self["sum_accuracy"] = value

    @property
    def sum_loss(self) -> float:
        return self.get("sum_loss", 0.0)

    @sum_loss.setter
    def sum_loss(self, value):
        self["sum_loss"] = value

    @property
    def count_rows(self) -> int:
        return self.get("count_rows", 0)

    @count_rows.setter
    def count_rows(self, value):
        self["count_rows"] = value

    @property
    def total_batch_num(self) -> int:
        return self.get("total_batch_num", 0)

    @total_batch_num.setter
    def total_batch_num(self, value):
        self["total_batch_num"] = value

    @property
    def epoch_batch_count(self) -> int:
        return self.get("epoch_batch_count", 0)

    @epoch_batch_count.setter
    def epoch_batch_count(self, value):
        self["epoch_batch_count"] = value


@dataclass
class Stats:
    accuracy_pct: float
    loss_average: float
    runtime_seconds: int
    steps: int
    dataset: str
    metadata: TrainingMetadata

    def to_json(self) -> dict:
        return asdict(self)


# Uploads files in background to not block the training loop.
class StorageBoxCheckpoint(StorageBox):
    def __init__(self, host, username, password, run_id, max_workers=2):
        today = date.today().isoformat()
        self.run_id = run_id
        self.base_name = f"checkpoints/{today}/{run_id}"
        super().__init__(host, username, password, self.base_name)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        atexit.register(self._shutdown)
        self._futures: list[Future] = []
        # debug flag
        self.sync = True

    def _submit(self, fn, *args) -> Future:
        if self.sync:
            # Run directly, wrap result in a completed Future
            f = Future()
            try:
                result = fn(*args)
                f.set_result(result)
            except Exception as e:
                f.set_exception(e)
            return f
        else:
            future = self._executor.submit(fn, *args)
            self._track(future)
            return future

    def checkpoint(self, model, optimizer, config, stats: Stats) -> Future:
        full_path = os.path.join(self.base_name, f"step_{stats.steps}")
        files = {
            "model.pt": self._serialize_torch(model),
            "optimizer.pt": self._serialize_torch(optimizer),
            "stats.json": self._serialize_json(stats),
            "config.json": self._serialize_json(config),
        }

        return self._submit(self._upload, files, full_path, stats)

    def tag(self, tag_name: str, stats: Stats):
        """Tag the current run so it can be found later by name."""
        tag_path = f"checkpoints/tags/{tag_name}"
        tag_data = {
            "run_id": self.run_id,
            "step": stats.steps,
            "path": os.path.join(self.base_name, f"step_{stats.steps}"),
            "tagged_at": datetime.now().isoformat(),
        }
        files = {"latest.json": self._serialize_json(tag_data)}
        return self._submit(self._upload, files, tag_path, stats)

    def checkpoint_files(self, files: Dict[str, bytes], stats: Stats) -> Future:
        full_path = os.path.join(self.base_name, f"step_{stats.steps}")
        serialized_files = {}
        for key, value in files.items():
            if isinstance(value, torch.nn.Module):
                serialized_files[key] = self._serialize_torch(value)
            else:
                serialized_files[key] = self._serialize_torch(value)

        future = self._executor.submit(self._upload, serialized_files, full_path, stats)
        self._track(future)
        return future

    def _handle_upload_error(self, future):
        exc = future.exception()
        if exc:
            print("Checkpoint upload failed", exc_info=exc)

    def _upload(self, files, full_path, stats: Stats):
        self.append_index_entry(self.run_id, stats.steps, full_path, stats.to_json())
        for name, data in files.items():
            try:
                self.save_bytes(data, os.path.join(full_path, name))
            except Exception as e:
                print(e)

    def _serialize_json(self, data) -> bytes:
        if isinstance(data, dict):
            raw = json.dumps(data, indent=2).encode("utf-8")
        else:
            raw = json.dumps(data.to_json(), indent=2).encode("utf-8")
        return raw

    def _serialize_torch(self, torch_state) -> bytes:
        buffer = io.BytesIO()
        torch.save(torch_state.state_dict(), buffer)
        return gzip.compress(buffer.getvalue(), compresslevel=1)

    def _track(self, future: Future) -> Future:
        self._futures.append(future)
        future.add_done_callback(lambda f: self._futures.remove(f))
        future.add_done_callback(self._handle_upload_error)
        return future

    def flush(self, timeout=None):
        if self.sync:
            return
        print("FLUSHING")
        for future in list(self._futures):
            print(f"Awaiting future {future}")
            future.result(timeout=timeout)
            print("DONE!")

    def _shutdown(self):
        self.flush()
        self._executor.shutdown(wait=True)


if __name__ == "__main__":
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )

    # Example: index existing checkpoints
    for filepath in storage.walk("checkpoints"):
        if not filepath.endswith("stats.json"):
            continue

        parts = filepath.split("/")
        run_id = parts[2]
        step = int(parts[3].replace("step_", ""))
        path = "/".join(parts[:-1])
        try:
            stats = json.loads(storage.load_bytes(filepath).decode("utf-8"))
        except Exception as e:
            print(e)
            continue
        storage.append_index_entry(run_id, step, path, stats)
        print(f"Indexed {run_id} step {step}")

    storage.close()
