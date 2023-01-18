from optimization_utils.envs.GridWorld import GridWorld
import matplotlib.pyplot as plt
import random
import os

def play_grid_world(agent, dirname="."):
    env = GridWorld(n=4)
    agent = agent(env.action_space)
    epochs = 5_000

    agent_y = []
    accumulated = 0
    for epochs in range(epochs):
        agent.train(env)
        
        if env.is_done():
            accumulated += 1

        env.reset()
        
        agent_y.append(accumulated)

        if epochs % 1_000 == 0:
            print(epochs)

    random_y = []
    accumulated = 0
    for epochs in range(epochs):
        while not env.done:
            env.play(random.sample(env.legal_actions, k=1)[0])
        
        if env.is_done():
            accumulated += 1

        random_y.append(
            accumulated
        )

        env.reset()

        if epochs % 1_000 == 0:
            print(epochs)

    plt.plot(agent_y, label=agent.__class__.__name__)
    plt.plot(random_y, label="Random actions")
    plt.title("Accumulated reward in grid world")
    plt.legend(loc="upper left")
    plt.savefig(
        os.path.join(
            dirname,
            agent.__class__.__name__ + '_grid_world.png'
        )
    )
