from utils.checkpoints import StorageBox
import os
from dotenv import load_dotenv
import json
from typing import Optional
from dataclasses import dataclass
import torch
from utils.web_dataloader import WebDataloader
from flask import Flask, request, jsonify
from utils.load_mode_from_checkpoint import (
    load_best_model_from_checkpoint,
    load_model_from_path,
)
from functools import lru_cache
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    argmax_sampling,
)

load_dotenv()


app = Flask(__name__)


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
        if self.accuracy < accuracy:
            self.accuracy = accuracy
            self.path = path


@lru_cache(maxsize=4)
def load_model_and_dataloader(
    target_dataset, model_path=None, dataloader_dataset=None, max_age_days=None
):
    if dataloader_dataset is None:
        dataloader_dataset = target_dataset
    if model_path is None:
        model, _ = load_best_model_from_checkpoint(
            target_dataset=target_dataset, max_age_days=max_age_days
        )
    else:
        model, _ = load_model_from_path(model_path)
    dataloader = WebDataloader(
        os.environ["WEB_DATALOADER"],
        dataloader_dataset,
        batch_size=1024,
    )
    return model, dataloader


@app.route("/embedding", methods=["POST"])
def embedding():
    print("Hello!")
    data = request.json
    text = data["text"]
    dataset = data["dataset"]
    method = data.get("method", "mean")
    normalize = data.get("normalize", False)
    max_age_days = data.get("max_age_days", 3)

    model, dataloader = load_model_and_dataloader(
        dataset, data.get("model_path"), max_age_days=max_age_days
    )
    doc_tensors = dataloader.tokenize([text])
    embeddings = torch.concat([model.embed(v) for v in doc_tensors], dim=0)

    if method == "mean":
        pooled = torch.mean(embeddings, dim=0)
    elif method == "max":
        pooled = torch.max(embeddings, dim=0).values
    elif method == "first":
        pooled = embeddings[0]
    elif method == "last":
        pooled = embeddings[-1]
    elif method == "weighted_decay":
        weights = torch.arange(len(embeddings), 0, -1, dtype=torch.float)
        weights = weights / weights.sum()
        pooled = (embeddings * weights.unsqueeze(1)).sum(dim=0)
    else:
        return jsonify({"error": f"Unknown method: {method}"}), 400

    if normalize:
        pooled = torch.nn.functional.normalize(pooled, dim=0)

    return jsonify(
        {
            "embedding": pooled.tolist(),
            "method": method,
            "normalized": normalize,
            "num_chunks": len(doc_tensors),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    print("Hello!")
    data = request.json
    documents = data["documents"]
    dataset = data["dataset"]
    dataloader_dataset = data.get("dataloader_dataset")
    model, dataloader = load_model_and_dataloader(
        dataset, data.get("model_path"), dataloader_dataset
    )
    model_response = []
    for text in documents:
        doc_tensors = dataloader.tokenize([text])[0]
        doc_tensors = doc_tensors[0]
        model_temperature_sampling = model.generate(
            doc_tensors, 128, temperature_sampling
        )
        model_argmax_sampling = model.generate(doc_tensors, 128, argmax_sampling)
        model_response.append(
            {
                "model_temperature_sampling": dataloader.detokenize(
                    model_temperature_sampling
                ),
                "model_argmax_sampling": dataloader.detokenize(model_argmax_sampling),
            }
        )

    return jsonify(model_response)


@app.route("/list", methods=["POST"])
def list():
    print("Hello!")
    data = request.json
    model_response = []
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    max_age_days = data.get("max_age_days", 3)
    min_age_days = data.get("min_age_days", 0)
    target_dataset = data["target_dataset"]
    for filepath in storage.walk(max_age_days=max_age_days, min_age_days=min_age_days):
        if not os.path.basename(filepath) == "stats.json":
            continue
        try:
            file_content = storage.load_bytes(filepath)
            data = json.loads(file_content)
        except Exception as e:
            print(f"Failed to load file: {e}")
        if target_dataset is None or data["dataset"] != target_dataset:
            continue
        model_path = os.path.dirname(filepath)
        run_id = int(os.path.basename(os.path.dirname(model_path)))
        model_response.append(
            {
                "model_path": model_path,
                "accuracy_pct": data["accuracy_pct"],
                "dataset": data["dataset"],
                "steps": data["steps"],
                "run_id": run_id,
            }
        )
    model_response = sorted(model_response, key=lambda x: (x["run_id"], x["steps"]))

    return jsonify(model_response)


if __name__ == "__main__":
    app.run(port=1259)

# if __name__ == "__main__":
#    load_best_model_from_checkpoint()
