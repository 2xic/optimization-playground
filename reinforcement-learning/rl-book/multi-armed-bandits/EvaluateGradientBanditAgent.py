import numpy as np
from envs.StaticBandit import StaticBandit
from helper.Parameter import Parameter
from helper.PlotParameter import PlotParameter
from helper.Plot import plot_optimal_action
from GradientBanditAgent import GradientBanditAgent
"""
Parameters
"""
K = 10
EVALUATION_RUNS = 5_00
EPOCH_LENGTH = 1_000

parameters = [
    (PlotParameter(Parameter("(with baseline) lr ", 0.4),
     {"color": "#3b56f5", "alpha": 0.5}), True),
    (PlotParameter(Parameter("(with baseline) lr ", 0.1),
     {"color": "#3b56f5"}), True),
    (PlotParameter(Parameter("(without baseline) lr ", 0.4),
     {"color": "#f5940c", "alpha": 0.5}), False),
    (PlotParameter(Parameter("(without baseline) lr ", 0.1),
     {"color": "#f5940c"}), False),
]

for (plot_lr_parameter, use_baseline) in parameters:
    lr_parameter = plot_lr_parameter.parameter
    lr = lr_parameter.value
    for eval in range(EVALUATION_RUNS):
        agent = GradientBanditAgent(
            K=K,
            use_baseline=use_baseline,
            lr=lr
        )

        with lr_parameter as results:
            bandit = StaticBandit(K, base_line=4, scale=1)

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
                )
        if eval % 100 == 0:
            print(eval)

plot_optimal_action(list(map(lambda x: x[0], parameters)), "gradient_bandit")
