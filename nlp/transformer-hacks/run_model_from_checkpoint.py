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
        if self.accuracy < accuracy:
            self.accuracy = accuracy
            self.path = path


def load_model_and_dataloader():
    model = load_best_model_from_checkpoint(target_dataset="small-web")
    model = model.create_embedding_model()
    dataloader = WebDataloader(
        os.environ["WEB_DATALOADER"],
        "small-web",
        batch_size=1024,
    )
    #    doc_tensors = dataloader.tokenize(["hello world!" * 256, "hello world!" * 512])
    #    for _, doc_tensor in enumerate(doc_tensors):
    #        print(model.forward_flatten(doc_tensor))
    return model, dataloader


model, dataloader = load_model_and_dataloader()
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]
    doc_tensors = dataloader.tokenize([text])
    #    for _, doc_tensor in enumerate(doc_tensors):
    #        print(model.forward_flatten(doc_tensor))
    embedding = model.forward_flatten(doc_tensors[0])
    # embedding = model.transforms([text])
    embedding = embedding[0]
    return jsonify(
        {
            "embedding": embedding.tolist(),
        }
    )


if __name__ == "__main__":
    app.run(port=1245)

# if __name__ == "__main__":
#    load_best_model_from_checkpoint()
