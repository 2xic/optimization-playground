from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass, asdict
import json
import torch
import tempfile
import paramiko
import time

load_dotenv()


class StorageBox:
    def __init__(
        self, host: str, username: str, password: str, remote_dir: str = "checkpoints"
    ):
        self.remote_dir = remote_dir
        self.password = password

        self.transport = paramiko.Transport((host, 23))
        self.transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

        self.create_directory(self.remote_dir)

    def _path_exists(self, path: str) -> bool:
        try:
            self.sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    def save(self, local_path: str, name: str):
        self.create_directory(self.remote_dir)
        self.sftp.put(local_path, f"{self.remote_dir}/{name}")

    def load(self, name: str, local_path: str):
        self.sftp.get(f"{self.remote_dir}/{name}", local_path)

    def delete(self, name: str):
        self.sftp.remove(f"{self.remote_dir}/{name}")

    def list(self) -> list[str]:
        self.create_directory(self.remote_dir)
        return self.sftp.listdir(self.remote_dir)

    def close(self):
        self.sftp.close()
        self.transport.close()

    def create_directory(self, directory):
        if not self._path_exists(directory):
            self.sftp.mkdir(directory)


@dataclass
class Stats:
    accuracy: float
    loss: float
    runtime_seconds: int
    steps: int

    def to_json(self) -> str:
        return asdict(self)


class StorageBoxCheckpointer(StorageBox):
    def __init__(self, host, username, password, run_id):
        self.base_name = f"checkpoints/{run_id}/"
        super().__init__(host, username, password, self.base_name)

    def checkpoint(self, model, optimizer, config, stats: Stats):
        remote_dir = f"step_{stats.steps}"
        self.create_directory(os.path.join(self.base_name, remote_dir))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._store_torch(
                model,
                tmp_path,
                os.path.join(remote_dir, "model.pt"),
            )
            self._store_torch(
                optimizer,
                tmp_path,
                os.path.join(remote_dir, "optimizer.pt"),
            )
            # Config files
            self._store_json(
                stats,
                tmp_path,
                os.path.join(remote_dir, "stats.json"),
            )
            self._store_json(
                config,
                tmp_path,
                os.path.join(remote_dir, "config.json"),
            )

    def _store_json(self, data, tmp_path, output):
        path = os.path.join(tmp_path, output)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = data.to_json()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        assert os.path.isfile(path)
        print(os.stat(path))
        time.sleep(0.1)
        self.save(path, output)

    def _store_torch(self, torch_state, tmp_path, output):
        path = os.path.join(tmp_path, output)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            torch.save(torch_state.state_dict(), f)
            f.flush()
            os.fsync(f.fileno())
        assert os.path.isfile(path)
        time.sleep(0.1)
        self.save(path, output)


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
    print("Available checkpoints:", checkpointer.list())
    #    checkpointer.load("local_checkpoint2.pth", "downloaded_checkpoint.pth")
    checkpointer.close()
