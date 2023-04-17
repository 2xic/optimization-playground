from .double_q_learning import Double_Q_learning
from .q_learning import Q_learning
from .parameter_study import evaluate_agents
from .BiasExample import BiasExample

import numpy as np
import matplotlib.pyplot as plt

from helpers.State import State
from helpers.StateValue import StateValue

from .double_q_learning import Double_Q_learning
from .q_learning import Q_learning
from helpers.analysis.Parameter import Parameter
from multiprocessing import Process, Lock, Queue
import os

SAMPLES = 10_000
EPOCHS = 300

def get_agent_data(lock, agent, queue):
    # as defined in the example
    alpha = 0.1
    eps = 0.1
    gamma = 1

    sum_reward = Parameter("alpha", alpha)

    for sample in range(SAMPLES):
        env = BiasExample()
        model = agent(env.action_space, eps)
        model.q = State(env.action_space, value_constructor=lambda n: StateValue(n, initial_value=lambda: 0))

        model.alpha = alpha
        model.gamma = gamma

        with sum_reward as x:
            for _ in range(EPOCHS):
                env.reset()
                model.train(env)
                x.add_reward(((1 if env.is_left else 0)) * 100)
        lock.acquire()
        try:
            print(model.__class__.__name__, sample)
        finally:
            lock.release()
   # print(sum_reward.get_reward())
    queue.put({
        "x": list(range(EPOCHS)),
        "y": sum_reward.get_reward(),
        "label": model.__class__.__name__
    })


def evaluate_agents_bias(agents, name):
    lock = Lock()
    processes = []
    queue = Queue()
    for agent in agents:
        processes.append(Process(target=get_agent_data, args=(lock, agent, queue)))
        processes[-1].start()

    for i in processes:
        i.join()
        item = queue.get()
        plt.plot(item["x"], item["y"], label=item["label"])

    plt.legend(loc="upper left")
    plt.ylabel("% of actions to the left")
    plt.xlabel("Episodes")
    plt.ylim(0, 100)
    plt.title('Bias check in model (lower % is better)')
    plt.hlines(y=5, colors='aqua', linestyles='-', xmin=0, xmax=len(item["x"]), lw=2, label='Optimal')
    plt.savefig(os.path.join(
        os.path.dirname(__file__),
        name
    ))    

if __name__ == "__main__":
    if False:
        evaluate_agents(
            agents=[Double_Q_learning, Q_learning],
            name='tic_tac_toe_q_learning_vs_double_q_learning.png',
        )
    evaluate_agents_bias(
        agents=[
            lambda actions, eps: Double_Q_learning(actions, eps=eps, decay=1, initial_value=lambda: 0), 
            lambda actions, eps: Q_learning(actions, eps=eps, decay=1)
        ],
        name='bias_q_learning_vs_double_q_learning.png',   
    )
