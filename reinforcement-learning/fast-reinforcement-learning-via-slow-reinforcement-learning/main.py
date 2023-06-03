
from optimization_utils.envs.TicTacToe import TicTacToe
import torch
from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from optimization_playground_shared.utils.SimpleAverage import SimpleAverage
import numpy as np
from train import train, play_random_agent

if __name__ == "__main__":
    avg = SimpleAverage()
    random_avg = SimpleAverage()
    for i in range(10):
        avg.add(
            train(
                iteration=i
            )
        )
        random_avg.add(
            play_random_agent()
        )

        training_accuracy = SimplePlot()
        training_accuracy.plot(
            [
                LinePlot(y=avg.res(), legend="R^2", title="Reward for tic tac toe", x_text="Iteration", y_text="Reward over time", y_min=-10, y_max=10),
                LinePlot(y=random_avg.res(), legend="Random", title="Reward for tic tac toe", x_text="Iteration", y_text="Reward over time", y_min=-10, y_max=10),
            ],
        )
        training_accuracy.save("rewards.png")
