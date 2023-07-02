"""
From the paper 

https://arxiv.org/pdf/2305.14342.pdf
"""
import numpy as np

class Sophia:
    def __init__(self, theta_0):
        self.theta = theta_0
        self.m = 0
        self.v = 0
        
        self.lam = .1
        self.eps = 1e-6

        self.k = 10
        self.beta_1 = 0.9
        self.beta_2 = 0.95

        self.t = 0

#        self.lr = 0.01
        self.lr = 1e-3
        # p does control a lot
#        self.p = 0.01
        self.p = 20

        self.h = 0

    def step(self, f, f_dx):
        gradient = f_dx(self.theta)

        self.m = self.beta_1 * self.m + (
            1 - self.beta_1
        ) * gradient

        if self.t % self.k == 1:
            h_theta = self.hutchinson(f, f_dx)
            self.h = self.beta_2 * self.h + \
                (1 - self.beta_2) * h_theta

        # weight decay
        self.theta -= self.lr * self.lam * self.theta
        delta = self._clip(
            self.m / max(self.h, self.eps),
            self.p
        )
        self.theta = self.theta - self.lr * delta

        if self.t % 100 == 0:
            print(self.theta)
            pass
        self.t += 1
        
    # as defined in section for "Pre-coordinate clipping"
    def _clip(self, z, p):
        return max(min(z, p), -p)


    def hutchinson(self, f, f_dx):
        u = np.random.normal(0, 1)
        gradient = f_dx(self.theta)
        result = np.multiply(u, np.dot(gradient, u))
        
        return result

if __name__ == "__main__":
    f = lambda x: (x-2)**2
    f_dx = lambda x: 2 * (x - 2)

    sophia = Sophia(0)
    for i in range(30_000):
        sophia.step(
            f,
            f_dx
        )
    print(sophia.theta)
