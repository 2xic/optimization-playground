# algorithm 1 from https://arxiv.org/pdf/2405.15682
import numpy as np

class AdamScheduleFree:
    def __init__(self, theta_0) -> None:
        self.b1, self.b2 = 0.9, 0.999
        self.lr = 1e-3
        self.warm_up_steps = 1
        self.decay = 0 

        self.x = theta_0
        self.z = theta_0        
        self.v = 0
        self.c = 0

        self.t = 0

        self.theta = self.x

    def step(self, _f, f_dx):
        self.t += 1

        eps = 10e-8

        y_t = (1 - self.b1) * self.z + self.b1 * self.x
        gradient = f_dx(self.z)

        self.v = self.b2 * self.v + (1 - self.b2) * (gradient ** 2)
        v_norm = self.v / (1 - self.b2**self.t) 
        # Reduce LR ? 
        lr = self.lr * min(1, self.t / self.warm_up_steps)
        self.z = self.z - lr * gradient / (np.sqrt(v_norm) + eps) - lr * self.decay * y_t
        # todo: optimize this 
        self.c = (lr ** 2) / ((np.ones(self.t)*lr ** 2).sum())
        self.x = (1 - self.c) * self.x + self.c*self.z

        if self.t % 100 == 0:
            print(self.x)
        
        assert np.isnan(self.x) == False
        self.theta = self.x

if __name__ == "__main__":
    f = lambda x: (x-2)**2
    f_dx = lambda x: 2 * (x - 2)

    adam = AdamScheduleFree(0)
    for i in range(30_000):
        adam.step(
            f,
            f_dx
        )
    