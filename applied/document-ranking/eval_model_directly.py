from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn import svm
from dataset import get_dataset
from pipeline import Pipeline
from xgboost import XGBRegressor
from optimization_playground_shared.utils.ClassImbalanceSplitter import balance_classes
from torch_score_models.linear import ModelInterface
import aiohttp
import asyncio
import json
import requests

class EmbeddingModel:
    def __init__(self, model):
        self.model = model

    def fit_transforms(self, docs):
        return asyncio.run(self._gather_all(docs))

    def transforms(self, docs):
        return asyncio.run(self._gather_all(docs))
    
    async def _gather_all(self, docs):
        values = []
        index = 0
        batch = []
        batch_size = 8
        while index < len(docs):
            if len(batch) >= batch_size:
                values += await asyncio.gather(*batch)
                batch = []
                print(f"{index} / {len(docs)}")
            text = docs[index]
            batch.append(self._get_embedding(text))
            index += 1
        # Add all missing
        values += await asyncio.gather(*batch)
        return values

    async def _get_embedding(self, text):
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8081/predict", json={
                "text": text,
                "model": self.model,
            }) as response:
                html = await response.text()
                results = json.loads(html)["embedding"]
                assert results is not None
                return results
            
    @classmethod
    def get_models(self):
        return requests.get("http://localhost:8081/models").json()["models"]

def evaluation():
    X, y = get_dataset()
    X, y = balance_classes(X, y)
    X = [str(i) for i in X]
    assert type(X[0]) == str, type(X[0])
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    model_scores = {}
    for embedding_model_name in EmbeddingModel.get_models():
        embedding_model = EmbeddingModel(embedding_model_name)
        (X_train, X_test, y_train, y_test) = Pipeline(sample_size=1).transform(
            X_train_original, X_test_original, y_train_original, y_test_original, embedding_model
        )
        # TODO: Add more fancy models also
        models = [
            ModelInterface(),
            RandomForestRegressor(max_depth=2, random_state=0),
            RandomForestRegressor(max_depth=8, random_state=0),
            RandomForestRegressor(max_depth=4, random_state=0),
            svm.SVR(),
            XGBRegressor(),
            KMeans(n_clusters=2),
            BisectingKMeans(n_clusters=2),
        ]
        best_score = 0
        for model in models:
            config_name =  f"{model.__class__.__name__}"
            print((config_name))
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, list(map(lambda x: min(max(round(x), 0), 1), model.predict(X_test))))
            print(f"\t{model.__class__.__name__} -> accuracy: {accuracy}")
            best_score = max(best_score, accuracy)
            print("")
        model_scores[embedding_model_name] = best_score
    print(f"Best score")
    print(json.dumps(model_scores, indent=4))

if __name__ == "__main__":
    evaluation()
