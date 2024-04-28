import matplotlib.pyplot as plt
from .adam_optimizer import Adam
import numpy as np

class HuberRegression:
    def __init__(self) -> None:
        # looked at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
        self.regularization = 0.0001
        self.epsilon = 1.35

    def fit(self, X, y, n_iterations=100_000):
        self.x = X
        self.y = y

        self.std = np.std(X)

        self.optim = Adam(0)
        for _ in range(n_iterations):
            self.optim.step(self.cost, self.cost_deriv)
        self.a = self.optim.theta

    def huber_function(self, value):
        if np.abs(value) < self.epsilon:
            return value ** 2
        else:
            return 2 * self.epsilon * np.abs(value) - self.epsilon ** 2

    def deriv_huber_function(self, value):
        if np.abs(value) < self.epsilon:
            return 2 * value
        else:
            return self.epsilon * np.sign(value)

    # Huber loss
    def cost(self, value):
        sum_cost = []
        for i in range(len(self.x)):
            sum_cost.append((
                self.std + 
                self.huber_function(
                    (self.x[i] * value - self.y[i]) / self.std
                ) * self.std
            ) + self.regularization * np.abs(value) * 2**2)
        return np.mean(sum_cost)
    
    # Huber loss
    def cost_deriv(self, value):
        sum_cost = []
        for i in range(len(self.x)):
            sum_cost.append(
                -2 * self.deriv_huber_function(
                    (self.x[i] * value - self.y[i]) / self.std
                ) * self.std
            )
        return np.mean(sum_cost)
    
    def predict(self, x):
        # y = a*x + b
        return list(map(lambda entry: (self.a * entry), x))

if __name__ == "__main__":
    x = list(range(1, 5))
    y = list(map(lambda x: (x * 2), x))

    X_prediction = list((range(1, 10)))

    model = HuberRegression()
    model.fit(x, y)
    assert (model.a) == 2
