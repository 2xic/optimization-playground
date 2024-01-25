"""
One big issue with models is reproducibility ... 

This file is meant to be something super simple so you can easily validate if your changes actually make a difference...
"""

class EvaluationOverTime:
    def __init__(self) -> None:
        self.rewards_epochs = []
        self.n = 1

    def add(self, rewards_epochs):
        for index, i in enumerate(rewards_epochs):
            if index < len(self.rewards_epochs) :
                self.rewards_epochs[index] += (i - self.rewards_epochs[index]) * (
                    1 / (self.n + 1)
                )
                self.n += 1
            else:
                self.rewards_epochs.append(i)
