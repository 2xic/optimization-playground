"""
Kinda like https://ai.meta.com/blog/billion-scale-semi-supervised-learning/

We add a endpoint on each model (rollout) which takes a list of items and sort it back with scores. We sum the sores and then sort it again, highest score first.
"""
from ranknet import Model as RanknetModel
from listnet_list import Model as ListnetMOdel
from listnet import Model as ListnetPlainModel
from bolztrank import Model as BoltzrankModel
from flask import Flask, request, jsonify
from results import Input, Results
from dotenv import load_dotenv
load_dotenv()

from optimization_playground_shared.apis.openai_embeddings import OpenAiEmbeddings
import torch
from abc import abstractmethod
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
            RanknetModel(embeddings_size=1536).load().eval(),
            ListnetMOdel(embeddings_size=1536).load().eval(),
            ListnetPlainModel(embeddings_size=1536).load().eval(),
            BoltzrankModel(embeddings_size=1536).load().eval()
        ]
    
    def rank_documents(self, items):
        documents = []
        is_raw_input = False
        for _, doc in enumerate(items):
            if isinstance(doc, Input):
                documents.append(doc)
                is_raw_input = True
            else:
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
        if is_raw_input:
            return list(sorted(items, key=lambda x: inference[x.item_id], reverse=True)), inference
        else:
            return list(sorted(items, key=lambda x: inference[x["id"]], reverse=True)), inference

    def rank_documents_tensor(self, items: torch.Tensor):
        # Input will be tensors ... But I want to call them through the rank documents
        results = torch.zeros((items.shape[0], items.shape[1]))
        for batch_size in range(items.shape[0]):
            documents = [
                Input(
                    item_id=index,
                    item_tensor=items[batch_size][index].reshape((1, -1))
                )
                for index in range(items.shape[1])
            ]
            _, inference = self.rank_documents(documents)
            results[batch_size] = torch.nn.Softmax(dim=0)(
                torch.tensor([inference[i] for i in range(items.shape[1])]).float()
            )
        return results

app = Flask(__name__)
models = SemiSupervisedMachine()

@app.route("/rank", methods=["POST"])
def rank_documents():
    data = request.json
    assert type(data) == list
    # Expected to just be a list of documents
    results, _ = models.rank_documents(data)
    return jsonify(list(map(lambda x: x, results)))

if __name__ == "__main__":
    app.run(port=4232)
