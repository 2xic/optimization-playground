
class Metric:
    def __init__(self, name) -> None:
        self.Q = []
        self.N = []
        self.index = 0

    def add_value(self, value):
        if len(self.Q) <= self.index:
            self.Q.append(float(value))
            self.N.append(1.0)
        else:
            self.N[self.index] += 1
            estimated_avg = 1/(
                self.N[self.index]
            )
            self.Q[self.index] += estimated_avg * float(
                value - self.Q[self.index]  
            )
            self.index += 1
