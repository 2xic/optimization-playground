

class RunningAverage:
    def __init__(self) -> None:
        self.value = None
        self.n = 0

    def update(self, value):
        if self.n == 0:
            self.value = value
            self.n += 1
        else:
            self.n += 1
            self.value += (value - self.value) / self.n
        return self.value
