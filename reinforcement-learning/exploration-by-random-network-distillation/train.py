from model import DqnModel
from optimization_utils.envs.TicTacToe import TicTacToe
import torch
from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
from rnd import Rnd
from optimization_playground_shared.utils.SimpleAverage import SimpleAverage
import numpy as np

if __name__ == "__main__":
    avg_rnd = SimpleAverage()
    avg_non_rnd = SimpleAverage()
    for i in range(10):
        env = TicTacToe(n=5, is_auto_mode=True)
        model = DqnModel(env.action_space)
        
        rnd_rewards = Rnd(env).train(model)
        
        model = DqnModel(env.action_space)
        plain_rewards = Rnd(env, disable_rnd=True).train(model)

        avg_rnd.add(np.asarray(rnd_rewards))
        avg_non_rnd.add(np.asarray(plain_rewards))

    training_accuracy = SimplePlot()
    training_accuracy.plot(
        [
            LinePlot(y=avg_rnd.res(), legend="Q-learning with RND", title="Reward for tic tac toe", x_text="Iteration", y_text="Reward over time", y_min=-10, y_max=10),
            LinePlot(y=avg_non_rnd.res(), legend="Q-learning", title="Reward for tic tac toe", x_text="Iteration", y_text="Reward over time", y_min=-10, y_max=10),
        ],
    )
    training_accuracy.save("rewards.png")
