import numpy as np


class SoftmaxSoftPolicy:
    def __init__(self) -> None:
        pass

    def __call__(self, vector, legal_actions=None):
        if type(legal_actions) == list:
            for i in legal_actions:
                vector[i] = 0
        vector = np.asarray(vector)
        softmax = self.softmax(vector)

        action = np.random.choice(vector.shape[-1], 1, p=softmax)[0]

        return action

    def softmax(self, vector):
        vector_exp = np.exp(vector)
        vector_sum = np.sum(vector_exp)

        softmax = (
            vector_exp / vector_sum
        )
        return softmax
