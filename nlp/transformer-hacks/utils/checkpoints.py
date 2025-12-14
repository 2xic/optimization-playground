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

load_dotenv()


class StorageBox:
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
        self.create_directory_recursive(self.remote_dir)

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

    def _cache_key(self, path: str) -> str:
        stat = self.sftp.stat(path)
        key_str = f"{path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(key_str.encode()).hexdigest()


@dataclass
class Stats:
    accuracy: float
    loss: float
    runtime_seconds: int
    steps: int

    def to_json(self) -> dict:
        return asdict(self)


class StorageBoxCheckpointer(StorageBox):
    def __init__(self, host, username, password, run_id):
        today = date.today().isoformat()
        self.base_name = f"checkpoints/{today}/{run_id}"
        super().__init__(host, username, password, self.base_name)

    def checkpoint(self, model, optimizer, config, stats: Stats):
        full_path = os.path.join(self.base_name, f"step_{stats.steps}")

        self._store_torch(model, os.path.join(full_path, "model.pt"))
        self._store_torch(optimizer, os.path.join(full_path, "optimizer.pt"))
        self._store_json(stats, os.path.join(full_path, "stats.json"))
        self._store_json(config, os.path.join(full_path, "config.json"))

    def _store_json(self, data, remote_path: str):
        buffer = io.BytesIO()
        buffer.write(json.dumps(data.to_json(), indent=2).encode("utf-8"))
        self.save_bytes(buffer.getvalue(), remote_path)

    def _store_torch(self, torch_state, remote_path: str):
        buffer = io.BytesIO()
        torch.save(torch_state.state_dict(), buffer)
        self.save_bytes(buffer.getvalue(), remote_path)


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
