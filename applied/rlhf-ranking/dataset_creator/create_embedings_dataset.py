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
            winner_id = data["winner"]["winner_id"]
            winner_text = get_embeddings(winner_id)
            if winner_text is None:
                continue
            looser_id = data["looser"]["looser_id"]
            looser_text = get_embeddings(looser_id)
            if looser_text is None:
                continue
            winner_embeddings = model.get_embedding(winner_text)
            looser_embeddings = model.get_embedding(looser_text)

            dataset.append([winner_embeddings, looser_embeddings])
    with open("dataset.json", "w") as file:
        json.dump(dataset, file)

if __name__ == "__main__":
    build()
