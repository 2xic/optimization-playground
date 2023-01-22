from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
import random
import os
from optimization_utils.diagnostics.Diagnostics import Diagnostics
from helpers.analysis.Parameter import Parameter
import json

def play_tic_tac_toe(agent_instance, dirname="."):
    EVALUATION_RUNS = 5
    EPOCHS = 10_000

    env = TicTacToe(n=3, is_auto_mode=True)
    agent_parameter = Parameter(None, None)

    for _ in range(EVALUATION_RUNS):
        agent = agent_instance(env.action_space)
        agent_diagnostics = Diagnostics()

        with agent_parameter as p:
            accumulated_reward = 0
            for epoch in range(EPOCHS):
                agent.train(env)
                reward = 0 if env.winner is None else env.winner

                agent_diagnostics.reward(reward)
                agent_diagnostics.track_raw_metric("eps", agent.epsilon.eps)
                accumulated_reward += reward

                p.add_reward(accumulated_reward)

                env.reset()

                if epoch % 1_000 == 0:
                    agent_diagnostics.print(epoch)

    random_parameter = Parameter(None, None)
    for _ in range(EVALUATION_RUNS):
        with random_parameter as p:
            accumulated_reward = 0
            for epoch in range(EPOCHS):
                while not env.done:
                    env.play(random.sample(env.legal_actions, k=1)[0])

                reward = 0 if env.winner is None else env.winner
                accumulated_reward += reward
                p.add_reward(accumulated_reward)

                env.reset()

                if epoch % 1_000 == 0:
                    print(epoch)

    plt.plot(agent_parameter.get_reward(), label=agent_instance.__name__)
    plt.plot(random_parameter.get_reward(), label="Random actions")
    plt.title(
        f"Average accumulated reward for tic tac toe (epochs {EPOCHS}, runs {EVALUATION_RUNS})")
    plt.legend(loc="upper left")
    plt.savefig(
        os.path.join(
            dirname,
            agent_instance.__name__ + '_tic_tac_toe.png'
        )
    )

    with open(
            os.path.join(dirname, ".data", agent_instance.__name__ + ".json"), "w") as file:
        json.dump({
            "agent": agent_parameter.get_reward(),
            "random": random_parameter.get_reward()
        }, file)
