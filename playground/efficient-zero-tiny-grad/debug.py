
class Debug:
    def __init__(self) -> None:
        self.rewards = []
        self.loss = []
        self.epoch = 0

    def add(self, sum_loss, sum_reward):
        self.loss.append(sum_loss)
        self.rewards.append(sum_reward)
        self.epoch += 1

    def print(self):
        avg_loss_last_4 = self.loss[-4:] 
        avg_loss_last_4 = sum(avg_loss_last_4) / len(avg_loss_last_4)

        avg_reward_last_4 = self.rewards[-4:] 
        avg_reward_last_4 = sum(avg_reward_last_4) / len(avg_reward_last_4)

        print(f"Epoch {self.epoch} | Loss : {avg_loss_last_4} | Reward {avg_reward_last_4}")
        self.epoch += 1
