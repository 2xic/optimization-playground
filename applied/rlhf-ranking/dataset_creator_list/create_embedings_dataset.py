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

from optimization_playground_shared.apis.openai_embeddings import OpenAiEmbeddings

def get_embeddings(link_id):
    url = os.environ["host_link"]
    # Find the content si.
    full_link = url + f"/text/{link_id}"
    results = requests.get(full_link, cookies={
        "credentials": os.environ["auth_header"]
    }).text
    if len(results) < 500:
        return None
    return results

def build():
    model = OpenAiEmbeddings()
    dataset = []
    for i in tqdm(glob.glob("results/*.json")):
        with open(i, "r") as file:
            data = json.load(file)
            text = []
            for i in data:
                winner_id = i["id"]
                winner_text = get_embeddings(winner_id)
                if winner_text is None:
                    break
                text.append(winner_text)
            if len(text) != len(data):
                continue
            embeddings = []
            for doc_text in text:
                winner_embeddings = model.get_embedding(doc_text)
                embeddings.append(winner_embeddings)
            dataset.append(embeddings)
    with open("dataset.json", "w") as file:
        json.dump(dataset, file)

if __name__ == "__main__":
    build()
