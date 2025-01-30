"""
Let's use OpenAi for embeddings backend.
"""
import glob
from dotenv import load_dotenv
import json
from tqdm import tqdm
from optimization_playground_shared.apis.url_to_text import get_text

load_dotenv()

def build():
    rows = []
    for i in tqdm(glob.glob("results/*.json")):
        with open(i, "r") as file:
            data = json.load(file)
            batch_text = []
            batch_scores = []
            for index, i in enumerate(data):
                url = i["url"]
                item_score = i.get("score", len(data) - index)
                item_text = get_text(url, cache_only=True)
                if item_text is None:
                    break
                batch_text.append(item_text)
                batch_scores.append(item_score)
            if len(batch_text) != len(data):
                print("skipping ... ")
                continue
            item_score = {}
            for (url, score) in zip(batch_text, batch_scores):
                item_score[url] = score
            rows.append(list(sorted(
                batch_text,
                key=lambda x: item_score[x]
            )))

    with open("dataset_ranked_documents.json", "w") as file:
        json.dump(rows, file, indent=4)

if __name__ == "__main__":
    build()
