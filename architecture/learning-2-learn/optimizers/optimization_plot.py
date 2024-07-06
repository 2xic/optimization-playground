"""
Plot of 2d function optimization
"""
from simple_adam import Adam
from simple_sophia import Sophia
from simple_schedule_free_adam import AdamScheduleFree
import matplotlib.pyplot as plt
import numpy as np

#theta_0 = 0
theta_0 = 5
epochs = 5_000
track_points = epochs / 4
lr = 1e-3

target_value = 2
def f(x): return (x-2)**2
def f_dx(x): return 2 * (x - 2)

def compare(optimizers, plot_name):
    x = np.linspace(-5, 5)
    y = f(x)

    _, axes = plt.subplots(nrows=2, ncols=1,figsize=(5,6))

    for index, optimizer in enumerate(optimizers):
        color = [
            "red",
            "blue"
        ][index]
        X_op = []
        y_op = []
        error_plot = []
        print(optimizer.__class__.__name__)
        optimizer.lr = lr 
        for epoch in range(epochs):
            optimizer.step(
                f,
                f_dx
            )
            if epoch % track_points == 0 or (epoch == (epochs - 1)):
                X_op.append(optimizer.theta)
                y_op.append(f(optimizer.theta))
                error_plot.append(target_value - optimizer.theta)
        axes[0].scatter(X_op, y_op, label=f"{optimizer.__class__.__name__}", color=color)
        axes[1].plot(error_plot, label=f"{optimizer.__class__.__name__}", color=color)
        print("")
    axes[0].legend(loc="upper left")
    axes[0].plot(x, y)
    plt.savefig(plot_name)
    plt.clf()

if __name__ == "__main__":
    compare([
        Adam(theta_0),
        Sophia(theta_0),
    ], "adam_vs_sophia.png")
    compare([
        Adam(theta_0),
        AdamScheduleFree(theta_0),
    ], "adam_vs_schedule_free_adam.png")
