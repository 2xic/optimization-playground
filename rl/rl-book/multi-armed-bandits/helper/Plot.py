import matplotlib.pyplot as plt
from .Parameter import Parameter
from .PlotParameter import Parameter
from typing import List
import numpy as np

def plot_average_reward(results: List[Parameter], base_name):
    plt.clf()
    plt.cla()
    for i in results:
        eps = i.value
        rewards = i.metrics["reward"].Q
       # incremental_reward = [rewards[0]]
       # for index in range(1, len(rewards)):
       #     incremental_reward.append(rewards[index] + incremental_reward[index-1])

        plt.plot(rewards, label=f"{i.name} {eps}")
        plt.xlabel('Steps')
        plt.ylabel('Average rewards')
        plt.ylim(0, max(max(rewards), 1))

    plt.legend(loc="upper left")
    plt.savefig(f"{base_name}_average_reward.png")

def plot_optimal_action(results: List[Parameter], base_name):
    plt.clf()
    plt.cla()
    for i in results:
        parameter = i if isinstance(i, Parameter) else i.parameter
        config = {} if isinstance(i, Parameter) else i.config

        eps = parameter.value
        optimal_action = list(map(lambda x: x * 100, parameter.metrics["optimal_action"].Q))

        plt.plot(optimal_action, label=f"{parameter.name} {eps}", **config)

    plt.xlabel('Steps')
    plt.ylabel('Average optimal action (%)')
    plt.ylim(0, 100)
    plt.legend(loc="upper left")
    plt.savefig(f"{base_name}_average_optimal_action.png")
