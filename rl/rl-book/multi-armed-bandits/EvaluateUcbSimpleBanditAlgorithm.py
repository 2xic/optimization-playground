import numpy as np
from envs.StaticBandit import StaticBandit
from helper.Parameter import Parameter
from helper.Plot import plot_average_reward, plot_optimal_action
from SimpleBanditAgent import SimpleBanditAgent
from action_policy.Ucb import UpperConfidenceBound
from action_policy.Epsilon import EpsilonGreedy
"""
Parameters
"""
K = 10
EVALUATION_RUNS = 100
EPOCH_LENGTH = 1_000

parameters = [
    Parameter("ucb", 0.2),
    Parameter("eps", 0.1),
]

for eps_parameter in parameters:
    eps = eps_parameter.value
    for _ in range(EVALUATION_RUNS):
        agent = SimpleBanditAgent(
            K=K,
            policy=(
                UpperConfidenceBound(K, eps_parameter.value)
                if eps_parameter.name == "ucb"
                else
                EpsilonGreedy(K, eps_parameter.value)
            )
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
                    reward=reward
                )

#print(parameters[0].metrics["reward"].Q)

plot_average_reward(parameters, "eps_vs_ucb")
# plot_optimal_action(parameters[:1], "eps_vs_ucb")

