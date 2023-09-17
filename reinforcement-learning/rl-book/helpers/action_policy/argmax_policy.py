import numpy as np

class ArgmaxPolicy:
    def __init__(self) -> None:
        pass

    def __call__(self, vector, legal_actions):
        if type(legal_actions) == list:
            for i in range(len(vector)):
                if i not in legal_actions:
                    vector[i] = -1
        vector = np.asarray(vector)
        return np.argmax(vector)
