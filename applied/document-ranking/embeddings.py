from optimization_playground_shared.apis.openai import OpenAiEmbeddings
from optimization_playground_shared.apis.cache_embeddings import CacheEmbeddings
from optimization_playground_shared.apis.huggingface import HuggingfaceApi
import requests
import os
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
load_dotenv()
import time

class OpenAiEmbeddingsWrapper:
    def __init__(self, model) -> None:
        self.is_trained = False
        self.encoder = OpenAiEmbeddings(model)

    def fit_transforms(self, x):
        return self.transforms(x)

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        return self.transforms(x)
    
    def transforms(self, x):
        X = []
        for i in tqdm(x):
            X.append(self.encoder.get_embedding(i))
        return X
    
# https://huggingface.co/models?pipeline_tag=feature-extraction&sort=trending
class HuggingFaceWrapper:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model
        self.is_trained = False
        self.cache_handler = CacheEmbeddings()
        self.api = HuggingfaceApi()

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        return self.transforms(x)
    
    def transforms(self, x):
        X = []
        for i in tqdm(x):
            X.append(self._query(self.model_id, i))
        return X

    def _query(self, model_id, texts):
        return self.api.get_embeddings(model_id, texts)

class ClaudeWrapper:
    def __init__(self) -> None:
        self.model_id = "voyage-2"
        self.token = os.environ["VOYAGE_API_KEY"]
        self.is_trained = False
        self.cache_handler = CacheEmbeddings()

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        return self.transforms(x)
    
    def transforms(self, x):
        X = []
        for i in tqdm(x):
            X.append(self._query(self.model_id, i))
        print(X)
        return X

    def _query(self, model_id, texts):
        payload = {
            "model": model_id,
            "input": texts,
        }
        cache = self.cache_handler.load(**payload)
        embed = None
        if cache is not None:
            embed = cache
        else:
            api_url = f"https://api.voyageai.com/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            payload["input_type"] = "query"
            payload["truncation"] = True
            
            response = requests.post(api_url, headers=headers, json=payload)
            embed = response.json()
            self.cache_handler.save(
                payload,
                embed
            )
            time.sleep(1)
        data = embed["data"][0]["embedding"]
        assert type(data) == list, data
        return np.asarray(data).astype(np.float32)
