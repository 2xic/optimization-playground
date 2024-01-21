from torch import Tensor

class Debug:
    def __init__(self) -> None:
        self.rewards = []
        self.loss = []
        self.epoch = 0
        self.predicted_state = {}

    def add(self, sum_loss, sum_reward):
        self.loss.append(sum_loss)
        self.rewards.append(sum_reward)
        self.epoch += 1

    def store_predictions(
        self,
        reward_predictions: Tensor,
        encoded_state_predicted: Tensor,
    ):
        self.predicted_state = {
            "reward_predictions": reward_predictions.detach().numpy(),
            "encoded_state_predicted": encoded_state_predicted.detach().numpy(),
        }

    def print(self):
        n = 4
        avg_loss_last_4 = self.loss[-n:] 
        avg_loss_last_4 = sum(avg_loss_last_4) / len(avg_loss_last_4)

        avg_reward_last_4 = self.rewards[-n:] 
        avg_reward_last_4 = sum(avg_reward_last_4) / len(avg_reward_last_4)

        print(f"Epoch {self.epoch} | Loss (avg last {n}) : {avg_loss_last_4} | Reward (avg last {n}) {avg_reward_last_4} | Last reward {self.rewards[-1]}")
        print(self.predicted_state)
