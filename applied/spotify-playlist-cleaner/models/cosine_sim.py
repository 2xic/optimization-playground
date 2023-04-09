import numpy as np

class PointIndex:
    def __init__(self) -> None:
        self.index = -1
        self.value = float('inf')

    def distance(self, X, X_dot, id):
        X = np.asarray(X)
        X_dot = np.asarray(X_dot)
        sim_score = self.sim(X, X_dot)

        if sim_score < self.value:
            self.value = sim_score
            self.index = id

    def sim(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

class CosineSim:
    def __init__(self):
        self.mappings = {

        }

    def fit(self, X, Y):
        for x, y in zip(X, Y):
            if y not in self.mappings:
                self.mappings[y] = []
            self.mappings[y].append(x)
        return self
    
    def predict(self, X):
        y_pred = []
        for x in X:
            point = PointIndex()
            for class_id in self.mappings:
                for item in self.mappings[class_id]:
                    point.distance(x, item, class_id)
            y_pred.append(point.index)
        return y_pred
    
