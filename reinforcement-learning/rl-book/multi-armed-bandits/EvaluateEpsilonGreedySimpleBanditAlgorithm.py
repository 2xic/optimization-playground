import numpy as np
from envs.StaticBandit import StaticBandit
from helper.Parameter import Parameter
from helper.Plot import plot_average_reward, plot_optimal_action
from SimpleBanditAgent import SimpleBanditAgent
from helpers.action_policy.Epsilon import EpsilonGreedy
"""
Parameters
"""
K = 10
EVALUATION_RUNS = 2_000
EPOCH_LENGTH = 1_000

parameters = [
    Parameter("eps", 0),
    Parameter("eps", 0.01),
    Parameter("eps", 0.1)
]

for eps_parameter in parameters:
    eps = eps_parameter.value
    for _ in range(EVALUATION_RUNS):
        agent = SimpleBanditAgent(
            K=K,
            policy=EpsilonGreedy(K, eps=eps)
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
                    action=action, reward=reward
                )

print(agent.Q)
print(bandit.rewards)

print(f"Estimated best action {np.argmax(agent.Q)}")
print(f"Actual best action {np.argmax(bandit.rewards)}")

plot_average_reward(parameters, "simple")
plot_optimal_action(parameters, "simple")
