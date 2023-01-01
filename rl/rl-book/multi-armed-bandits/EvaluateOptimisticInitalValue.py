import numpy as np
from envs.StaticBandit import StaticBandit
from helper.Parameter import Parameter
from helper.Plot import plot_average_reward, plot_optimal_action
from SimpleBanditAgent import SimpleBanditAgent
from action_policy.Epsilon import EpsilonGreedy
"""
Parameters
"""
K = 10
EVALUATION_RUNS = 2_000
EPOCH_LENGTH = 1_000

parameters = [
    (Parameter("q_0=2.5 eps", 0), 2.5, 0.1),
    (Parameter("q_0=0.0 eps", 0.1), 0, 0.1),
]

for (eps_parameter, q_0, lr) in parameters:
    eps = eps_parameter.value
    for _ in range(EVALUATION_RUNS):
        agent = SimpleBanditAgent(
            K=K,
            policy=EpsilonGreedy(K, eps=eps),
            q_0=q_0
        )

        with eps_parameter as results:
            bandit = StaticBandit(K)
            for _ in range(EPOCH_LENGTH):
                action = agent.action()
                reward = bandit(action)

                results.add_reward(reward)
                results.add_lifetime_metric("optimal_action", int(
                    action == bandit.optimal_action
                ))

                agent.update(
                    action=action, 
                    reward=reward,
                    lr=lr
                )

plot_optimal_action(list(map(lambda x: x[0], parameters)), "optimistic_initial_value")

