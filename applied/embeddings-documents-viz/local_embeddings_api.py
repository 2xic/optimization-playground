import requests

class LocalEmbeddingsModelApi:
    def __init__(self, model) -> None:
        self.model = model

    def get_embedding(self, text):
        return self._request(text)

    def _request(self, text):
        response = requests.post(
            "http://localhost:1245/predict",
            json={
                "model": self.model,
                "text": text
            }
        ).json()
        print(response)
        
        return response["embedding"]

    def name(self):
        return self.model
