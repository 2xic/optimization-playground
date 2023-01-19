from .Metric import Metric

class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.metrics = {
            "reward": Metric("Reward")
        }

    def get_reward(self):
        return self.metrics["reward"].get_value()
        
    def add_reward(self, value):
        self.metrics["reward"].add_value(value)

    def add_lifetime_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = Metric(name)
        self.metrics[name].add_value(value)

    def __enter__(self) -> None:
        for i in self.metrics:
            self.metrics[i].index = 0
        return self

    def __exit__(self, *args) -> None:
        pass
