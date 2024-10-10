"""
Simple resource tracker

All in memory 
"""
from collections import defaultdict

class ResourceTracker:
    def __init__(self):
        self.cpu = [] # -> Consumer should report it in a way that we just plot the percentage
        self.ram = [] # 
        self.gpu = [] # 
        self.gpus = defaultdict(list)
        self.max_size = 50

    def add_usage(self, device, add_percentage):
        if device == "cpu":
            self.cpu.append(add_percentage)
        elif device == "ram":
            self.ram.append(add_percentage)
        elif device == "gpu":
            self.gpu.append(add_percentage)
        elif device == "gpus":
            for key, value in add_percentage.items():
                self.gpus[key].append(value)
                self.gpus[key] = self.gpus[key][-self.max_size:]
        else:
            raise Exception("Unknown device")

        self.cpu = self.cpu[-self.max_size:]
        self.gpu = self.gpu[-self.max_size:]
        self.ram = self.ram[-self.max_size:]
        print((device, add_percentage))
