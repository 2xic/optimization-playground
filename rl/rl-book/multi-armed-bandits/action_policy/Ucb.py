import numpy as np

class UpperConfidenceBound:
    def __init__(self, K, c) -> None:
        self.N = [0, ] * K
        self.c = c
        self.t = 0

    def __call__(self, agent):
        a_t = []
        for index, q_a in enumerate(agent.Q):
            if self.N[index] == 0:
                a_t.append(float('inf'))
            else:
                uncertainty = self.c * np.sqrt(np.log(self.t) / self.N[index])
            #    print(uncertainty)
                a_t.append(
                    q_a + \
                    uncertainty
                )
          #      print((q_a, uncertainty))

        action = np.argmax(a_t)

        self.N[action] += 1
        self.t += 1

        return action
