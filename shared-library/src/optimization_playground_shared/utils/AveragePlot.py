from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class AveragePlot:
    def __init__(self):
        self.labels = defaultdict(list)

    def add(self, label, y):
        self.labels[label].append(y)

    def plot(self, filename):
        for key, value in self.labels.items():
            value = np.asarray(value).sum(axis=0)/len(value)
            plt.plot(value, label=key)
        plt.legend(loc="upper left")
        plt.savefig(filename)
