import torch
import numpy as np
from scipy.spatial.distance import pdist, cdist


def check_embedding_geometry(frequent_embeddings, rare_embeddings):
    assert not np.array_equal(frequent_embeddings, rare_embeddings)
    # 1. Frequent words should be spread out (high pairwise distance)
    freq_distances = pdist(frequent_embeddings, metric="cosine")
    spread_score = np.mean(freq_distances)  # Higher = more spread out

    # 2. Rare words should be close to some frequent word (low min distance)
    rare_to_freq_distances = cdist(
        rare_embeddings, frequent_embeddings, metric="cosine"
    )
    min_distances = np.min(rare_to_freq_distances, axis=1)
    clustering_score = 1 - np.mean(min_distances)  # Higher = closer to frequent words

    return {
        "spread_score": spread_score,
        "clustering_score": clustering_score,
        "overall_score": (spread_score + clustering_score) / 2,
    }


class WordEmbedding:
    def __init__(self, dataset):
        skip_token = dataset.tokenizer.padding_index
        self.base_tensor = (
            torch.zeros((1, dataset.dataset.sequence_size)).fill_(skip_token).long()
        )
        self.dataset = dataset

    def word_embedding(self, word, model):
        self.base_tensor[0][0] = self.dataset.tokenizer.encode_idx(word)
        with torch.no_grad():
            output = model(self.base_tensor)[0]  # .mean(dim=1)
        #      print(output)
        return output


def test_model(dataset, model):
    top_100, lower_100 = dataset.get_token_distribution(100)
    word_embedding = WordEmbedding(dataset)

    top_embeddings = []
    low_embeddings = []
    for i in top_100:
        top_embeddings.append(word_embedding.word_embedding(i, model))
    for i in lower_100:
        low_embeddings.append(word_embedding.word_embedding(i, model))
    assert len(low_embeddings) == len(top_embeddings)

    top_embeddings = torch.concat(top_embeddings, dim=0)
    low_embeddings = torch.concat(low_embeddings, dim=0)
    #'print'((low_embeddings.shape, top_embeddings.shape))
    scores = check_embedding_geometry(
        top_embeddings.cpu().numpy(), low_embeddings.cpu().numpy()
    )
    print(f"Spread score: {scores['spread_score']:.3f}")
    print(f"Clustering score: {scores['clustering_score']:.3f}")
    print(f"Overall score: {scores['overall_score']:.3f}")
