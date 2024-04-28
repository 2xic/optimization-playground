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

    def step(self, _f, f_dx):
        self.t += 1

        eps = 10e-8

        gradient = f_dx(self.theta)
        self.m = self.b1 * self.m + (1- self.b1) * gradient
        self.v = self.b2 * self.v + (1 - self.b2) * gradient ** 2

        m_bias = self.m / (1 - self.b1)
        v_bias = self.v/ (1 - self.b2)

        self.theta = self.theta - self.lr * m_bias/(np.sqrt(v_bias) + eps)

        if self.t % 1_000 == 0:
            print(self.theta)
