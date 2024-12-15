"""
Web interface for running a ranker model.
"""
from flask import Flask, request
import os
from dotenv import load_dotenv
from ranknet import Model
from semi_supervised_machine import Input
import torch
from dataset_creator_list.embedding_backend import EmbeddingBackend

results_dirname = "results"
os.makedirs(results_dirname, exist_ok=True)

load_dotenv()

app = Flask(__name__)

embeddings = EmbeddingBackend()
model = Model(
    embeddings_size=1536,
).load()

@app.route('/', methods=["POST"])
def index():
    """
    Input is text and output is the 
    """
    docs = request.json["documents"]
    documents = []
    for i in docs:
        documents.append(Input(
            item_id=i["id"],
            item_tensor=torch.tensor([embeddings.get_embedding(i["text"])]),
        ))
    results = model.rollout(documents)
    items = []
    for i in results:
        items.append({
            "id": i.item_id,
            "score": i.item_score.item(),
        })
    return items

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2343)
