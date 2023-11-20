"""
Simple resource tracker

All in memory 
"""

class ResourceTracker:
    def __init__(self):
        self.cpu = [] # -> Consumer should report it in a way that we just plot the percentage
        self.ram = [] # 
        self.gpu = [] # 
        self.max_size = 50

    def add_usage(self, device, add_percentage):
        if device == "cpu":
            self.cpu.append(add_percentage)
        elif device == "ram":
            self.ram.append(add_percentage)
        else:
            self.gpu.append(add_percentage)

        self.cpu = self.cpu[-self.max_size:]
        self.gpu = self.gpu[-self.max_size:]
        self.ram = self.ram[-self.max_size:]
        print((device, add_percentage))
