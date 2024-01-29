

class LossDebug:
    def __init__(self) -> None:
        self.loss_reward = []
        self.loss_policy = []
        self.loss_projection_loss = []

    def add(self, policy, reward, projection_loss):
        self.loss_policy.append(policy)
        self.loss_reward.append(reward)
        self.loss_projection_loss.append(projection_loss)
