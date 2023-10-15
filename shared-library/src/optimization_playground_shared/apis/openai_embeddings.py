"""
Get the 
"""
import os
import requests
from dotenv import load_dotenv
import json
from .cache_embeddings import CacheEmbeddings
load_dotenv()

cache_handler = CacheEmbeddings()
token = os.getenv("OPEN_AI_API_KEY")

assert token is not None or 0 < len(token), "Api key not found :("

def _save_json(name, data):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        name
    )
    with open(path, "w") as file:
        file.write(json.dumps(data))

def get_models():
    results = requests.get("https://api.openai.com/v1/models", 
        headers={
            "Authorization": f"Bearer {token}" 
        }
    )
    _save_json(
        "models.json",
        results.json()
    )

def get_embeddings(text, model):
    payload = {
            # TODO: Improve the limit checks
            "input": " ".join(text.split(" ")[:3_000]),
            "model": model
    }
    cache = cache_handler.load(**payload)
    if cache is not None:
        return cache
    
  #  print(payload)
    results = requests.post("https://api.openai.com/v1/embeddings?model", 
        json=payload,
        headers={
            "Authorization": f"Bearer {token}" 
        }
    )
    #print(results)
    cache_handler.save(
        payload=payload,
        results=results.json()
    )
    return results.json()
