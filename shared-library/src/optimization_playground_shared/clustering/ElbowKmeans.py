from .elbow_method import ElbowMethod
from sklearn.cluster import KMeans

class ElbowKmeans:
    def __init__(self) -> None:
        self.elbow_method = ElbowMethod()
        self.model = None

    def fit(self, X, _y):
        clusters = self.elbow_method.fit(X)
        self.model = KMeans(n_clusters=clusters)
        return self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)
    