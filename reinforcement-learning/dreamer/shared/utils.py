
class RewardPrinter:
    def __init__(self) -> None:
        self.rewards = []

    def add_reward(self, reward):
        self.rewards.append(reward)
        self.rewards = self.rewards[-25:]

    def print(self):
        min_r = min(self.rewards)
        max_r = max(self.rewards)
        avg_r = sum(self.rewards) / len(self.rewards)
        last_r = self.rewards[-1]
        return (f"min: {min_r}, max: {max_r}, avg: {avg_r}, last: {last_r}")


