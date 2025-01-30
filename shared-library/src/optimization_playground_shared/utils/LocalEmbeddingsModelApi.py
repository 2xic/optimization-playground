import requests
from tqdm import tqdm

class LocalEmbeddingsModelApi:
    def __init__(self, model, host) -> None:
        self.model = model
        self.host = host

    def get_embedding(self, text):
        return self._request(text)

    def fit_transforms(self, x):
        return self.transforms(x)

    def transforms(self, x):
        X = []
        for i in tqdm(x):
            X.append(self._request(i))
        return X

    def _request(self, text):
        response = requests.post(
            f"{self.host}/predict",
            json={
                "model": self.model,
                "text": text
            }
        )
        response = response.json()        
        return response["embedding"]

    def name(self):
        return self.model
