from .double_q_learning import Double_Q_learning
from .q_learning import Q_learning
from .sarsa import Sarsa
from .tabluar_td_0 import Tabluar_td_0
from .expected_sarsa import ExpectedSarsa
from optimization_utils.envs.TicTacToe import TicTacToe
import numpy as np
import matplotlib.pyplot as plt
from helpers.analysis.Parameter import Parameter
from multiprocessing import Process, Lock, Queue
import os

SAMPELS = 100
EPOCHS = 500

def get_agent_data(lock, agent, queue):
    X = []
    y = []

    for alpha in np.linspace(0, 1, 5):
        env = TicTacToe()
        model = agent(env.action_space)
        model.alpha = alpha

        sum_reward = Parameter("alpha", alpha)
        for sample in range(SAMPELS):
            with sum_reward as x:
                sum_reward_value = 0
                for _ in range(EPOCHS):
                    env.reset()
                    sum_reward_value += model.train(env)
                x.add_reward(sum_reward_value)
            lock.acquire()
            try:
                print(agent.__name__, sample, sum_reward.get_reward())
            finally:
                lock.release()

        X.append(alpha)
        y.append(sum_reward.get_reward()[0])
    #plt.plot(X, y, label=agent.__name__)
    queue.put({
        "x": X,
        "y": y,
        "label": agent.__name__
    })

lock = Lock()
processes = []
queue = Queue()
for agent in [Double_Q_learning, Q_learning, Sarsa, Tabluar_td_0, ExpectedSarsa]:
    processes.append(Process(target=get_agent_data, args=(lock, agent, queue)))
    processes[-1].start()

for i in processes:
    i.join()
    item = queue.get()
    plt.plot(item["x"], item["y"], label=item["label"])

plt.legend(loc="upper left")
plt.ylabel("Reward")
plt.xlabel("Alpha")
plt.title('Algorithms alhpa parameter')
plt.savefig(os.path.join(
    os.path.dirname(__file__),
    'alpha_parameter_search.png'
))
