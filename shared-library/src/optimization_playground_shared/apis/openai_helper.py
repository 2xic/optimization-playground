"""
Get the 
"""
import os
import requests
from dotenv import load_dotenv
import json
from .cache_embeddings import CacheEmbeddings
from typing import Dict
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
    # TODO: TOken length validation is in OpenAiAdaEmbeddings
    payload = {
            "input": text,
            "model": model
    }
    cache = cache_handler.load(**payload)
    if cache is not None and "error" not in cache:
        return cache
    
    print("Requesting embedding ... ")
    results = requests.post("https://api.openai.com/v1/embeddings?model", 
        json=payload,
        headers={
            "Authorization": f"Bearer {token}" 
        }
    )

    cache_handler.save(
        payload=payload,
        results=results.json()
    )
    return results.json()

def get_completion(messages, model, response_format=None) -> Dict:
    # TODO: TOken length validation is in OpenAiAdaEmbeddings
    payload = {
        "model": model,
        "messages": messages,
     #   "max_completion_tokens": 50_000
    }
    print(model)
    if response_format is not None:
        payload["response_format"] = response_format
    cache = cache_handler.load(
        model,
        input=json.dumps(payload)
    )
    if cache is not None and "error" not in cache:
        return cache
    
    print("Requesting embedding ... ")
    results = requests.post("https://api.openai.com/v1/chat/completions", 
        json=payload,
        headers={
            "Authorization": f"Bearer {token}" 
        }
    )

    cache_handler.save(
        payload={
            "model":model,
            "input": json.dumps(payload),
        },
        results=results.json()
    )
    return results.json()

def get_text_to_speech(text):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": text,
            "voice": "alloy"
        },
        headers={
            "Authorization": f"Bearer {token}" 
        },
        stream=True
    )
    return response
