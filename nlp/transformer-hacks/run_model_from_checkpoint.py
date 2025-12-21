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


def load_best_model_from_checkpoint():
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    best_model_path = BestModelResult(path="checkpoints/2025-12-14/")
    queue = [None]
    while len(queue) > 0:
        base = queue.pop()
        for i in storage.list(base):
            print(i)
            if storage.is_directory(i):
                queue.append(i)
            elif os.path.basename(i) == "stats.json":
                data = json.loads(storage.load_bytes(i))
                print(data)
                best_model_path.update_by_accuracy(
                    data["accuracy_pct"], os.path.dirname(i)
                )

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


model, dataloader = load_best_model_from_checkpoint()
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
