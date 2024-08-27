"""
Kinda like https://ai.meta.com/blog/billion-scale-semi-supervised-learning/

We add a endpoint on each model (rollout) which takes a list of items and sort it back with scores. We sum the sores and then sort it again, highest score first.
"""
from ranknet import Model as RanknetModel
from listnet_list import Model as ListnetMOdel
from flask import Flask, request, jsonify
from results import Input, Results
from dotenv import load_dotenv
load_dotenv()

from optimization_playground_shared.apis.openai_embeddings import OpenAiEmbeddings
import torch
from abc import ABCMeta, abstractmethod
from typing import List
from collections import defaultdict

class RankingModel:
    @abstractmethod
    def rollout(self, items: List[Input]) -> List[Results]:
        pass
 
class SemiSupervisedMachine:
    def __init__(self) -> None:
        self.embeddings = OpenAiEmbeddings()
        self.models: List[RankingModel] = [
            RanknetModel(embeddings_size=1536).load(),
            ListnetMOdel(embeddings_size=1536).load(),
        ]
    
    def rank_documents(self, items):
        documents = []
        for _, doc in enumerate(items):
            documents.append(Input(
                item_id=doc["id"],
                item_tensor=torch.tensor([self.embeddings.get_embedding(doc["description"])])
            ))
        inference = defaultdict(list)
        for model in self.models:
            results = model.rollout(documents)
            for _, item in enumerate(results):
                inference[item.item_id].append(item.item_score)
        for result in inference:
            inference[result] = sum(inference[result])
        # Then resort it again :) based on biggest score
        return list(sorted(items, key=lambda x: inference[x["id"]], reverse=True))

app = Flask(__name__)
models = SemiSupervisedMachine()

@app.route("/rank", methods=["POST"])
def rank_documents():
    data = request.json
    assert type(data) == list
    # Expected to just be a list of documents
    results = models.rank_documents(data)
    return jsonify(list(map(lambda x: x, results)))

if __name__ == "__main__":
    app.run(port=4232)
