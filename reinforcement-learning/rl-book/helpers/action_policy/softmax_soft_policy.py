import numpy as np


class SoftmaxSoftPolicy:
    def __init__(self) -> None:
        pass

    def __call__(self, vector, legal_actions=None):
        action_values = []
        if type(legal_actions) == list:
            for j in range(0, len(vector)):
                if j in legal_actions:
                    action_values.append(vector[j])
        if len(action_values) != 0:
            vector = action_values
        vector = np.asarray(vector)
        softmax = self.softmax(vector)

        action = np.random.choice((
            vector.shape[-1] if legal_actions is None else legal_actions
        ), 1, p=softmax)[0]
       # print(action)

        return action

    def softmax(self, vector):
        vector_exp = np.exp(vector)
        vector_sum = np.sum(vector_exp)

        softmax = (
            vector_exp / vector_sum
        )
        return softmax
