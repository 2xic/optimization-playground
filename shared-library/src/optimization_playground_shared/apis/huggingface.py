import time
import requests
from .cache_embeddings import CacheEmbeddings
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

class HuggingfaceApi:
    def __init__(self) -> None:
        self.hf_token = os.environ["hugginf_face"]
        self.cache_handler = CacheEmbeddings()

    def get_summary(self, model_id, texts):
        payload = {
            "model": model_id,
            "input": texts,
        }
        cache = self.cache_handler.load(**payload)
        if cache is not None:
            return cache
        # 20:42
        print(payload)
        api_url = f"https://api-inference.huggingface.co/pipeline/summarization/{model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        response = requests.post(
            api_url,
            headers=headers, 
            json={"inputs": texts, "options":
                {
                    "wait_for_model":True
                }
            }
        )
        embed = response.json()
        if type(embed) == dict:
            print(embed)
        time.sleep(1)
        return response

    def get_embeddings(self, model_id, texts):
        payload = {
            "model": model_id,
            "input": texts,
        }
        cache = self.cache_handler.load(**payload)
        if cache is not None:
            return cache
        # 20:42

        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        embed = response.json()
        if type(embed) == dict:
            print(embed)
        response = np.asarray(embed).astype(np.float32)
        self.cache_handler.save(
            payload,
            embed
        )
        time.sleep(1)
        return response
