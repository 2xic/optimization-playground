import matplotlib.pyplot as plt
from .adam_optimizer import Adam
import numpy as np

class ProbabilisticLinearRegression:
    def __init__(self) -> None:
        pass

    def fit(self, X, y, n_iterations=100_000):
        self.x = X
        self.y = y

        # Mean square error
        self.cost = lambda a: 1/2 * np.mean(([(self.y[i] - a * self.x[i]) ** 2 for i in range(len(self.y))]))
        self.cost_deriv = lambda a: -2 * np.mean(sum([(self.y[i] - a * self.x[i]) for i in range(len(self.y))]))

        self.optim = Adam(0)
        for _ in range(n_iterations):
            self.optim.step(self.cost, self.cost_deriv)
        self.a = self.optim.theta

    """
    y = ax + b
    """
    def plot(self, x, output):
        assert type(x) == list
        plt.plot(self.x, self.y, label="Measured", color="blue")
        plt.plot(x, list(map(lambda x: self.a * x, x)), label="Predicted", linestyle="dotted", color="blue")
        plt.legend(loc="upper left")
        plt.savefig(output)
        plt.clf()

    def predict(self, x):
        # y = a*x + b
        print("rela a", self.a)
        return list(map(lambda entry: (self.a * entry), x))

if __name__ == "__main__":
    x = list(range(1, 5))
    y = list(map(lambda x: (x * 2), x))

    X_prediction = list((range(1, 10)))

    model = ProbabilisticLinearRegression()
    model.fit(x, y)
    assert (model.a) == 2

    model.plot(X_prediction, "simple_linear.png")
