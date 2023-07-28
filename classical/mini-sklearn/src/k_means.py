import numpy as np

"""
Same example as in https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
"""

class KMeans:
    def __init__(self, n_clusters) -> None:
        self.n_clusters = n_clusters
        self.centers = None

    def fit(self, X, eps=1e-6):
        self.centers = np.random.rand(
            *(self.n_clusters, ) + X.shape[1:]
        ) * 5
        for _ in range(1_000):
            closest_centers = np.zeros(
                (self.n_clusters, ) + X.shape[1:]
            )
            closest_centers_count = np.zeros((
                self.n_clusters
            ))
            for index in range(len(X)):
                element = X[index]
                closest_index = self._find_closest_centroid(element)
                closest_centers[closest_index] += element
                closest_centers_count[closest_index] += 1

            self.centers = closest_centers / (closest_centers_count + eps)
            
        return self

    def predict(self, X):
        if type(X) == list:
            X = np.asarray(X)
        predicted = np.zeros((
            X.shape[0]
        ))
        for index in range(len(X)):
            predicted[index] = self._find_closest_centroid(X[index])
            
        return predicted

    def _find_closest_centroid(self, element):
        closest_index = ((self.centers - element) ** 2).sum(axis=1)
        closest_index = closest_index.argmin(axis=0)
        return closest_index

if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(
        n_clusters=2
    ).fit(X)
    print(kmeans.predict([[0, 0], [12, 3]]))
    # array([1, 0], dtype=int32)
    