import zmq
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

"""
We could also do some partial caching on this dataloader
"""
class ZmqDataloader(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        host = os.environ["ZMQ_HOST"]
        assert ":" not in host, "Expected host to not be defined"
        self.socket.connect(f"tcp://{host}:5555")
        # 5 second timeout on sending and receive
        self.socket.setsockopt(zmq.SNDTIMEO, 5_000)
        self.socket.setsockopt(zmq.RCVTIMEO, 5_000)
        self.documents = {}

    def __len__(self):
        self.socket.send_json({
            "command": "size"
        })
        print("SENT")
        message = self.socket.recv()
        print(message)
        return int(message)

    # You want to do fancy stuff, you do it here.
    def process_message(self, message):
        return message.replace(b"\r\n", b"\n")

    def __getitem__(self, idx):
        # TODO cache based on idx ?
        try:
            if idx in self.documents:
                return self.documents[idx]
            self.socket.send_json({
                "command": "get",
                "arguments": {
                    "index": idx,
                }
            })
            message = self.socket.recv()
            self.documents[idx] = self.process_message(message)
            processed_message = self.documents[idx]
            return processed_message
        except zmq.Again as e:
            return self.__getitem__(idx)

def get_dataloader():
    train_ds = ZmqDataloader()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    return train_loader


if __name__ == "__main__":
    dataloader = get_dataloader()
    for index, X in enumerate(dataloader):
        for y in range(len(X)):
            print(f"{index}_{y}", len(X[y]), hashlib.sha256(X[y]).hexdigest())

