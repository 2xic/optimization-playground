"""
Simple implementation of the elbow method
"""
from sklearn.cluster import KMeans

class ElbowMethod:
    def __init__(self) -> None:
        pass

    def fit(self, X):
        clusters = 1
        last_error = float('inf')
        while True:
            k_means = KMeans(n_clusters=clusters)
            k_means.fit(
                X
            )
            error = k_means.inertia_
            delta = (last_error - error ) / error 
            # "elbow" likely
            if delta < 0.1:
                break
            last_error = error
            clusters += 1
        return clusters
    