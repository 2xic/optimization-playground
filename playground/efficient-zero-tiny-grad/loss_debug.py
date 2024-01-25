

class LossDebug:
    def __init__(self) -> None:
        self.loss_reward = []
        self.loss_policy = []

    def add(self, policy, reward):
        self.loss_policy.append(policy)
        self.loss_reward.append(reward)

