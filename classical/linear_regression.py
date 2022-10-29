

# algorithm 1 from https://arxiv.org/pdf/1412.6980.pdf
import numpy as np

class Adam:
    def __init__(self, theta_0) -> None:
        self.b1, self.b2 = 0.9, 0.999
        self.lr = 1e-3
        
        self.m = 0
        self.v = 0
        self.t = 0

        self.theta = theta_0

    def step(self, f, f_dx):
        self.t += 1

        eps = 10e-8

        gradient = f_dx(self.theta)
        self.m = self.b1 * self.m + (1- self.b1) * gradient
        self.v = self.b2 * self.v + (1 - self.b2) * gradient ** 2

        m_bias = self.m / (1 - self.b1)
        v_bias = self.v/ (1 - self.b2)

        self.theta = self.theta - self.lr * m_bias/(np.sqrt(v_bias) + eps)

        if self.t % 100 == 0:
            print(self.theta)


class ProbabilisticLinearRegression:
    def __init__(self) -> None:
        self.x = list(range(1, 5))
        self.y = list(map(lambda x: x * 2, self.x))
        self.cost = lambda a: (1/2 * sum([(self.y[i] - a * self.x[i]) ** 2 for i in range(len(self.y))]))
        self.cost_deriv = lambda a:  (-2 * sum([(self.y[i] - a * self.x[i]) for i in range(len(self.y))]))
        self.optim = Adam(0)
        for i in range(100000):
            self.optim.step(self.cost, self.cost_deriv)
        self.a = self.optim.theta
    """
    y = ax + b
    """
    def plot(self):
        pass


if __name__ == "__main__":
    x = ProbabilisticLinearRegression()
    assert (x.a) == 2

