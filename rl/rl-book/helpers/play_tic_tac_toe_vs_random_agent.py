from optimization_utils.envs.TicTacToe import TicTacToe
import matplotlib.pyplot as plt
import random
import os


def play_tic_tac_toe(agent, dirname="."):
    env = TicTacToe(n=4, is_auto_mode=True)
    agent = agent(env.action_space)
    epochs = 10_000

    agent_y = []
    accumulated = 0
    for epochs in range(epochs):
        agent.train(env)
        reward = 0 if env.winner is None else env.winner
        accumulated += reward

        agent_y.append((
            accumulated
        ))

        env.reset()

        if epochs % 1_000 == 0:
            print(epochs)

    random_y = []
    accumulated = 0
    for epochs in range(epochs):
        while not env.done:
            env.play(random.sample(env.legal_actions, k=1)[0])

        reward = 0 if env.winner is None else env.winner
        accumulated += reward

        random_y.append(
            accumulated
        )

        env.reset()

        if epochs % 1_000 == 0:
            print(epochs)

    plt.plot(agent_y, label=agent.__class__.__name__)
    plt.plot(random_y, label="Random actions")
    plt.title("Accumulated reward for tic tac toe")
    plt.legend(loc="upper left")
    plt.savefig(
        os.path.join(
            dirname,
            agent.__class__.__name__ + '_tic_tac_toe.png'
        )
    )
