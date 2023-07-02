"""
Math from https://d2l.ai/chapter_optimization/rmsprop.html
"""
import numpy as np

class RMSprop:
    def __init__(self, theta_0) -> None:
        self.b1, self.b2 = 0.9, 0.999
        self.lr = 1e-3
        self.eps = 10e-6
        self.gamma = 0.7

        self.s = 0

        self.t = 0
        self.theta = theta_0

    def step(self, f, f_dx):
        self.t += 1

        gradient = f_dx(self.theta)

        self.s = self.gamma * self.s + (1 - self.gamma) * gradient ** 2
        self.theta = self.theta - self.lr / np.sqrt(self.s + self.eps) * gradient

        if self.t % 100 == 0:
            print(self.theta)

if __name__ == "__main__":
    def f(x): return (x-2)**2
    def f_dx(x): return 2 * (x - 2)

    rmsprop = RMSprop(0)
    for i in range(30_000):
        rmsprop.step(
            f,
            f_dx
        )
    