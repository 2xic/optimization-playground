from .double_q_learning import Double_Q_learning
from .q_learning import Q_learning
from .parameter_study import evaluate_agents
from .BiasExample import BiasExample

import numpy as np
import matplotlib.pyplot as plt

from .double_q_learning import Double_Q_learning
from .q_learning import Q_learning
from helpers.analysis.Parameter import Parameter
from multiprocessing import Process, Lock, Queue
import os
from helpers.plot_compress import savefig

SAMPELS = 1_00
EPOCHS = 300

def get_agent_data(lock, agent, queue):
    alpha = 0.1
    eps = 0.1
    gamma = 1

    sum_reward = Parameter("alpha", alpha)
    for sample in range(SAMPELS):
        env = BiasExample()
        model = agent(env.action_space, eps)
        model.alpha = alpha
        model.gamma = gamma

        with sum_reward as x:
            for _ in range(EPOCHS):
                env.reset()
                model.train(env)
                x.add_reward((
                    (1 if env.is_left else 0) * 100
                ))
        lock.acquire()
        try:
            print(model.__class__.__name__, sample)
     #      print(sum_reward.get_reward())
        finally:
            lock.release()
  #  

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
    savefig(os.path.join(
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
        name='bias_q_learning_vs_double_q_learning.jpg',   
    )
