import zmq
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import os
import json

load_dotenv()

"""
We could also do some partial caching on this dataloader
"""
class ZmqDataloader(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        host = os.environ["ZMQ_HOST"]
        assert ":" not in host, "Expected host to not be defined"
        self.socket.connect(f"tcp://{host}:5555")

    def __len__(self):
        self.socket.send_json({
            "command": "size"
        })
        message = self.socket.recv()
        return int(message)

    # You want to do fancy stuff, you do it here.
    def process_message(self, message):
        return message

    def __getitem__(self, idx):
        # TODO cache based on idx ? 
        self.socket.send_json({
            "command": "get",
            "arguments": {
                "index": idx,
            }
        })
        message = self.socket.recv()
        return self.process_message(message)

def get_dataloader():
    train_ds = ZmqDataloader()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    return train_loader


if __name__ == "__main__":
    for X in get_dataloader():
        print(X)

