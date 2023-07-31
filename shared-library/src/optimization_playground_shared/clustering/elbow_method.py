"""
Simple implementation of the elbow method
"""
from sklearn.cluster import KMeans
import numpy as np
import time
class ElbowMethod:
    def __init__(self) -> None:
        pass

    def fit(self, X):
        clusters = 1
        last_error = float('inf')
        while True:
            kmeans = KMeans(n_clusters=clusters)
            kmeans.fit(
                X
            )
            error = kmeans.inertia_
            delta = (last_error - error ) / error 
            # "elbow" likely
            if delta < 0.1:
                break
            last_error = error
            clusters += 1
            #time.sleep(0.4)
        return clusters
    