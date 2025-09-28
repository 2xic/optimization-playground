import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ConfidenceInterval:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @classmethod
    def create(cls, value):
        return cls(count=1, mean=value, m2=0.0)

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        return self

    @property
    def variance(self):
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def std(self):
        return self.variance**0.5

    @property
    def lower_bound(self):
        return self.mean - self.std

    @property
    def upper_bound(self):
        return self.mean + self.std


class MinMaxAvgArray:
    def __init__(self):
        self.min_max_avg: List[ConfidenceInterval] = []

    def add(self, entries):
        is_new = len(self.min_max_avg) == 0
        # assert is_new or len(self.min_max_avg) == len(entries)
        for index, i in enumerate(entries):
            if is_new:
                self.min_max_avg.append(ConfidenceInterval.create(i))
            else:
                self.min_max_avg[index].update(i)

    def get_arrays(self):
        min = list(map(lambda x: x.lower_bound, self.min_max_avg))
        max = list(map(lambda x: x.upper_bound, self.min_max_avg))
        avg = list(map(lambda x: x.mean, self.min_max_avg))
        return min, max, avg


@dataclass
class Results:
    accuracy: MinMaxAvgArray
    loss: MinMaxAvgArray


def running_average(data):
    running_sum = 0
    for i, value in enumerate(data, 1):
        running_sum += value
        if i > 0:
            running_avg = running_sum / i
        yield running_avg


def plot_accuracy_loss(results: Dict[str, Results], file_path: str):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    # Plot Accuracy
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    colors = ["blue", "green", "red", "yellow", "orange", "purple"]
    for index, (key, value) in enumerate(results.items()):
        (min, max, avg) = value.accuracy.get_arrays()
        x = np.arange(len(min))
        # avg = list(running_average(avg))
        # min = list(running_average(min))
        # max = list(running_average(max))

        #  print((min, max, avg))

        ax1.plot(x, avg, color=colors[index], label=f"Accuracy ({key})", alpha=0.6)
        ax1.fill_between(x, min, max, color=colors[index], alpha=0.2)

    ax1.tick_params(axis="y")
    ax1.legend(loc="lower right")
    ax1.set_title("Accuracy")

    # Plot Loss
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    for index, (key, value) in enumerate(results.items()):
        (min, max, avg) = value.loss.get_arrays()
        avg = list(running_average(avg))
        min = list(running_average(min))
        max = list(running_average(max))
        x = np.arange(len(min))

        ax2.plot(x, avg, color=colors[index], label=f"Loss ({key})", alpha=0.6)
        ax2.fill_between(x, min, max, color=colors[index], alpha=0.2)
    ax2.tick_params(axis="y")
    ax2.legend(loc="upper right")
    ax2.set_title("Loss")

    file_path = file_path.split(".")[0]
    print(f"Output: {file_path}.png")
    plt.savefig(f"{file_path}.png")
    plt.close("all")
