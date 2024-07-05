from sklearn.feature_extraction.text import TfidfVectorizer
from optimization_playground_shared.apis.openai_ada_embeddings import OpenAiAdaEmbeddings
from optimization_playground_shared.apis.cache_embeddings import CacheEmbeddings
import requests
import os
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
load_dotenv()
import time

class TfIdfWrapper:
    def __init__(self, **kwargs) -> None:
        self.encoder = TfidfVectorizer(**kwargs)
        self.is_trained = False

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        return self.encoder.fit_transform(x)
    
    def transforms(self, x):
        return self.encoder.transform(x)

class OpenAiEmbeddingsWrapper:
    def __init__(self) -> None:
        self.is_trained = False
        self.encoder = OpenAiAdaEmbeddings()

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        return self.transforms(x)
    
    def transforms(self, x):
        X = []
        for i in x:
            X.append(self.encoder.get_embedding(i))
        return X
    
class HuggingFaceWrapper:
    def __init__(self) -> None:
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.hf_token = os.environ["hugginf_face"]
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
        return X

    def _query(self, model_id, texts):
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
