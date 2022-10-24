"""
algorithm 2 from https://arxiv.org/pdf/1412.6980.pdf

Simple ADamax
"""


class AdaMax:
    def __init__(self, theta_0) -> None:
        self.b1, self.b2 = 0.9, 0.999
        self.lr = 1e-3

        self.m = 0
        self.u = 0
        self.t = 0

        self.theta = theta_0

    def step(self, f, f_dx):
        self.t += 1

        gradient = f_dx(self.theta)
        self.m = self.b1 * self.m + (1 - self.b1) * gradient
                                        # vector norm
        self.u = max(self.b2 * self.u,  gradient ** 2 )

        self.theta = self.theta - (self.lr / (1 - self.b1)) * self.m/self.u

        if self.t % 100 == 0:
            print(self.theta)


if __name__ == "__main__":
    def f(x): return (x-2)**2
    def f_dx(x): return 2 * (x - 2)

    adam = AdaMax(0)
    for i in range(30_000):
        adam.step(
            f,
            f_dx
        )
