"""
Let's use OpenAi for embeddings backend.
"""
import glob
import requests
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm

load_dotenv()

from optimization_playground_shared.apis.openai import OpenAiEmbeddings

def get_embeddings(link_id):
    url = os.environ["host_link"]
    # Find the content si.
    full_link = url + f"/text/{link_id}?refetch=true"
    results = requests.get(full_link, cookies={
        "credentials": os.environ["auth_header"]
    }).text
    if len(results) < 100:
        full_link = url + f"/url/{link_id}"
        results = requests.get(full_link, cookies={
            "credentials": os.environ["auth_header"]
        }).text
        print(f"Skipping {link_id} ({results})")
        return None
    return results

def build():
    model = OpenAiEmbeddings()
    dataset = []
    for i in tqdm(glob.glob("results/*.json")):
        with open(i, "r") as file:
            data = json.load(file)
            text = []
            scores = []
            for index, i in enumerate(data):
                item_id = i["id"]
                item_score = i.get("score", len(data) - index)
                item_text = get_embeddings(item_id)
                if item_text is None:
                    break
                text.append(item_text)
                scores.append(item_score)
            if len(text) != len(data):
                continue
            # Embeddings
            embeddings = []
            for doc_text in text:
                winner_embeddings = model.get_embedding(doc_text)
                embeddings.append(winner_embeddings)
            dataset.append({
                "embeddings": embeddings,
                "scores": scores,
            })
    with open("dataset.json", "w") as file:
        json.dump(dataset, file)

if __name__ == "__main__":
    build()
