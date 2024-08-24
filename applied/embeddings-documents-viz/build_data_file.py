from dataset import get_dataset
from optimization_playground_shared.apis.openai_embeddings import OpenAiEmbeddings
import hashlib
import json
import numpy as np
from tqdm import tqdm
from local_embeddings_api import LocalEmbeddingsModelApi
"""
1. Get the dataset.
2. Inference the models on the datasets
{
    [model-name]: {
        [docId]:{
            "embeddings": [
                "<embeddings-data>"
            ],
            "metadata": {
                "url": "<link>
            }
        }
    }
}
3. Run the t-sne model to compress to 2d points.
"""
from sklearn.manifold import TSNE

def get_point(prediction):
    tsne = TSNE(random_state=1, n_components=2, n_iter=15000, metric="cosine", init='random', learning_rate='auto')
    embs = tsne.fit_transform(prediction)
    print(embs.shape)
    # Point coordinates
    return embs * 100

def document_id(text):
    return hashlib.sha256(text.encode()).hexdigest()

def build_dataset_file():
    X, _, metadata = get_dataset()
    results = {}
    for model in [
        LocalEmbeddingsModelApi("gpt_like_model"),
        OpenAiEmbeddings("text-embedding-ada-002"),
        OpenAiEmbeddings("text-embedding-3-large"),
        OpenAiEmbeddings("text-embedding-3-small"),
    ]:
        embeddings = []
        embeddings_metadata = []
        results[model.name()] = {}
        for index, document in enumerate(tqdm(X)):
            embedding = np.asarray(model.get_embedding(document))
            embeddings.append(
                embedding
            )
            embeddings_metadata.append(
                {
                    "id": document_id(document),
                    "url": metadata[index]["url"]
                }
            )
        # Output is the embeddings
        points = get_point(np.asarray(embeddings))
        for index, (embeddings_metadata, point) in enumerate(zip(embeddings_metadata, points)):
            results[model.name()][embeddings_metadata["id"]] = {
                "point": {
                    "x": int(point[0]),
                    "y": int(point[1])
                },
                "url": embeddings_metadata["url"]
            }
    # Save it
    with open("embeddings.json", "w") as file:
        json.dump(results, file)

if __name__ == "__main__":
    build_dataset_file()

