from utils.checkpoints import StorageBox
import os
from dotenv import load_dotenv
import json
from typing import Optional
from dataclasses import dataclass
from experiments import Config, Model
import torch
import io
from utils.web_dataloader import WebDataloader
from flask import Flask, request, jsonify
from utils.load_mode_from_checkpoint import load_best_model_from_checkpoint
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
def load_model_and_dataloader(target_dataset, dataloader_dataset=None):
    if dataloader_dataset is None:
        dataloader_dataset = target_dataset
    model, _ = load_best_model_from_checkpoint(
        target_dataset=target_dataset, max_age_days=3
    )
    dataloader = WebDataloader(
        os.environ["WEB_DATALOADER"],
        dataloader_dataset,
        batch_size=1024,
    )
    return model, dataloader


@app.route("/embedding", methods=["POST"])
def embedding():
    data = request.json
    text = data["text"]
    dataset = data["dataset"]
    method = data.get("method", "mean")
    normalize = data.get("normalize", False)

    model, dataloader = load_model_and_dataloader(dataset)
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
    data = request.json
    documents = data["documents"]
    dataset = data["dataset"]
    dataloader_dataset = data["dataloader_dataset"]
    model, dataloader = load_model_and_dataloader(dataset, dataloader_dataset)
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


if __name__ == "__main__":
    app.run(port=1255)

# if __name__ == "__main__":
#    load_best_model_from_checkpoint()
