from utils.checkpoints import StorageBox
import os
from dotenv import load_dotenv
import json
from typing import Optional
from dataclasses import dataclass
from training.model import Config, Model
import torch
import io
from typing import Tuple

load_dotenv()


@dataclass
class BestModelResult:
    loss: Optional[int] = None
    accuracy: Optional[int] = None
    path: Optional[str] = None

    def update_by_loss(self, loss, path):
        if self.loss is None or loss < self.loss:
            self.loss = loss
            self.path = path

    def update_by_accuracy(self, accuracy, path):
        if self.accuracy is None or self.accuracy < accuracy:
            self.accuracy = accuracy
            self.path = path


def load_best_model_from_checkpoint(target_dataset) -> Tuple[Model, Config]:
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    best_model_path = BestModelResult()
    queue = [None]
    while len(queue) > 0:
        base = queue.pop()
        for i in storage.list(base):
            print(i)
            if storage.is_directory(i):
                queue.append(i)
            elif os.path.basename(i) == "stats.json":
                data = json.loads(storage.load_bytes(i))
                if data["dataset"] != target_dataset:
                    continue
                best_model_path.update_by_accuracy(
                    data["accuracy_pct"], os.path.dirname(i)
                )
    if best_model_path.path is None:
        raise Exception("No model found")
    model_config = Config.from_json(
        json.loads(
            storage.load_bytes(os.path.join(best_model_path.path, "config.json"))
        )
    )
    print(best_model_path)
    print(model_config)
    model = Model(model_config)
    print(model)
    print("Loading weights ... ")
    weights = torch.load(
        io.BytesIO(storage.load_bytes(os.path.join(best_model_path.path, "model.pt"))),
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(weights)
    return (model, model_config)
