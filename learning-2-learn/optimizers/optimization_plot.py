"""
Plot of 2d function optimization
"""
from simple_adam import Adam
from simple_sophia import Sophia
import matplotlib.pyplot as plt
import numpy as np

theta_0 = 0
epochs = 1_000
track_points = epochs / 4


optimizer = [
    Adam(theta_0),
    Sophia(theta_0)
]


def f(x): return (x-2)**2
def f_dx(x): return 2 * (x - 2)


x = np.linspace(-5, 5)
y = f(x)

for index, i in enumerate(optimizer):
    color = [
        "red",
        "blue"
    ][index]
    X_op = []
    y_op = []
    for epoch in range(epochs):
        i.step(
            f,
            f_dx
        )
        if epoch % track_points == 0 or (epoch == (epochs - 1)):
            X_op.append(i.theta)
            y_op.append(f(i.theta))
    plt.scatter(X_op, y_op, label=f"{i.__class__.__name__}", color=color)

plt.legend(loc="upper left")
plt.plot(x, y)
plt.savefig('optimization.png')
