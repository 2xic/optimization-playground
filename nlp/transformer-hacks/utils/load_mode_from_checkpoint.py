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


def load_model_from_path(best_model_path):
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    model_config = Config.from_json(
        json.loads(storage.load_bytes(os.path.join(best_model_path, "config.json")))
    )
    print(best_model_path)
    print(model_config)
    model = Model(model_config)
    print(model)
    print("Loading weights ... ")
    weights = torch.load(
        io.BytesIO(storage.load_bytes(os.path.join(best_model_path, "model.pt"))),
        map_location=torch.device("cpu"),
    )
    new_state_dict = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(new_state_dict)
    print("Model loaded!")
    return (model, model_config)


def load_best_model_from_checkpoint(
    target_dataset, max_age_days=3
) -> Tuple[Model, Config]:
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    best_model_path = BestModelResult()

    for filepath in storage.walk(max_age_days=max_age_days):
        if os.path.basename(filepath) == "stats.json":
            try:
                file_content = storage.load_bytes(filepath)
                data = json.loads(file_content)
            except Exception as e:
                print(f"Failed to load file: {e}")
            print(data["dataset"])
            if data["dataset"] == target_dataset:
                best_model_path.update_by_accuracy(
                    data["accuracy_pct"], os.path.dirname(filepath)
                )
    if best_model_path.path is None:
        raise Exception("No model found")
    return load_model_from_path(best_model_path.path)
