"""
Let's use OpenAi for embeddings backend.
"""
import glob
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
from optimization_playground_shared.apis.url_to_text import get_text, get_url_documents

load_dotenv()

from embedding_backend import EmbeddingBackend

def build():
    model = EmbeddingBackend()
    scores = []
    text = []
    flatten_text = []

    dataset = []
    for i in tqdm(glob.glob("results/*.json")):
        with open(i, "r") as file:
            data = json.load(file)
            batch_text = []
            batch_scores = []
            for index, i in enumerate(data):
                #item_id = i["id"]
                url = i["url"]
                item_score = i.get("score", len(data) - index)
                item_text = get_text(url)
                if item_text is None:
                    break
                batch_text.append(item_text)
                batch_scores.append(item_score)
                flatten_text.append(item_text)
            if len(batch_text) != len(data):
                print("skipping ... ")
                continue
            text.append(batch_text)
            scores.append(batch_scores)
            print(len(scores))
    # Train model
    if hasattr(model.transformer, "train"):
        # load on the input documents + some extra for good measure.
        extra_documents = get_url_documents()
        extra_documents += flatten_text
        model.transformer.train(extra_documents)

    assert len(scores) > 0
    assert len(text) == len(scores)

    for (batch_scores, batch_docs) in zip(scores, text):
        # Embeddings
        embeddings = []
        for doc_text in batch_docs:
            winner_embeddings = model.get_embedding(doc_text)
            embeddings.append(winner_embeddings)
        dataset.append({
            "embeddings": embeddings,
            "scores": batch_scores,
        })
    with open(f"dataset_{model.backend}.json", "w") as file:
        json.dump(dataset, file)

if __name__ == "__main__":
    build()
