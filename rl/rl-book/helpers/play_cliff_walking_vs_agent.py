from optimization_utils.envs.CliffWalking import CliffWalking
import matplotlib.pyplot as plt
import os
from optimization_utils.diagnostics.Diagnostics import Diagnostics
from helpers.analysis.Parameter import Parameter
from .random_agents import RandomAgent

EVALUATION_RUNS = 30
EPOCHS = 500

def evaluate_agent(env, agent_instance):
    agent_parameter = Parameter(None, None)

    for _ in range(EVALUATION_RUNS):
        agent = agent_instance(env.action_space, eps=0.1, decay=1)
        agent_diagnostics = Diagnostics()

        with agent_parameter as p:
            for epoch in range(EPOCHS):
                reward = agent.train(env)

                agent_diagnostics.reward(reward)
                p.add_reward(reward)

                env.reset()

                if epoch % 1_000 == 0:
                    agent_diagnostics.print(epoch)
    return agent_parameter
    
def play_cliff_walking(first_agent_instance, second_agent_instance=RandomAgent, dirname="."):
    env = CliffWalking(n=3)

    first_agent_parameter = evaluate_agent(env, first_agent_instance)
    second_agent_parameter = evaluate_agent(env, first_agent_instance)

    plt.plot(first_agent_parameter.get_reward(), label=first_agent_instance.__name__)
    plt.plot(second_agent_parameter.get_reward(), label=second_agent_instance.__name__)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of reward in episode')
    plt.title(
        f"Average accumulated reward for cliff walking (epochs {EPOCHS}, runs {EVALUATION_RUNS})")
    plt.legend(loc="upper left")
    plt.savefig(
        os.path.join(
            dirname,
            first_agent_instance.__name__ + "_" + second_agent_instance.__name__ + '_cliff_walking.png'
        )
    )
