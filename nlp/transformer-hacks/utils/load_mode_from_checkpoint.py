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


def load_model_from_path(base_model_path):
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    model_config = Config.from_json(
        json.loads(storage.load_bytes(os.path.join(base_model_path, "config.json")))
    )
    print(base_model_path)
    print(model_config)
    model = Model(model_config)
    print(model)
    print("Loading weights ... ")
    weights = torch.load(
        io.BytesIO(storage.load_bytes(os.path.join(base_model_path, "model.pt"))),
        map_location=torch.device("cpu"),
    )
    new_state_dict = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(new_state_dict)
    print("Model loaded!")
    return (model, model_config)


def load_raw_from_path(base_model_path):
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    stats = json.loads(storage.load_bytes(os.path.join(base_model_path, "stats.json")))
    model_state = torch.load(
        io.BytesIO(storage.load_bytes(os.path.join(base_model_path, "model.pt"))),
        map_location=torch.device("cpu"),
    )
    optimizer_state = torch.load(
        io.BytesIO(storage.load_bytes(os.path.join(base_model_path, "optimizer.pt"))),
        map_location=torch.device("cpu"),
    )
    return model_state, optimizer_state, stats


def iterate_over_dataset(target_dataset, max_age_days):
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    #    for filepath in storage.walk(max_age_days=max_age_days):
    #        if os.path.basename(filepath) == "stats.json":
    for entry in storage.iterate_index():
        dataset = entry["stats"]["dataset"]
        filepath = entry["path"]
        if dataset == target_dataset:
            yield filepath, entry["stats"]


def load_best_model_from_checkpoint(
    target_dataset, max_age_days=3
) -> Tuple[Model, Config]:
    best_model_path = BestModelResult()
    for model_dir, data in iterate_over_dataset(target_dataset, max_age_days):
        # TODO: was some older entries which had a corrupted values, they can probably soon be disregarded.
        if data["accuracy_pct"] <= 100:
            best_model_path.update_by_accuracy(data["accuracy_pct"], model_dir)
    if best_model_path.path is None:
        raise Exception("No model found")
    print(best_model_path.path)
    return load_model_from_path(best_model_path.path)
