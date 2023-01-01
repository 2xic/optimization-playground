from envs.StaticBandit import StaticBandit
from SimpleBanditAgent import SimpleBanditAgent
from helper.ParameterStudy import ParameterStudy
from helper.Parameter import Parameter
from GradientBanditAgent import GradientBanditAgent
from action_policy.Epsilon import EpsilonGreedy
from action_policy.Ucb import UpperConfidenceBound
import matplotlib.pyplot as plt
from helper.RunningAverage import RunningAverage

"""
Parameters
"""
K = 10
EVALUATION_RUNS = 1_00
EPOCH_LENGTH = 1_000

parameter_agent = [
    ParameterStudy(
        "e-greedy", [1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1],
        apply_parameter=lambda eps: SimpleBanditAgent(
            K=K,
            policy=EpsilonGreedy(K, eps=eps)
        ),
        apply_update=lambda agent, action, reward: agent.update(
            action=action,
            reward=reward
        )
    ),
    ParameterStudy(
        "UCB", [1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1],
        apply_parameter=lambda ucb: SimpleBanditAgent(
            K=K,
            policy=UpperConfidenceBound(
                K=K,
                c=ucb
            )
        ),
        apply_update=lambda agent, action, reward: agent.update(
            action=action,
            reward=reward
        )
    ),
    ParameterStudy(
        "Greedy initialization", [1/8, 1/4, 1/2, 1, 2, 2.5, 3, 3.5],
        apply_parameter=lambda q_0: SimpleBanditAgent(
            K=K,
            policy=EpsilonGreedy(
                K,
                eps=0
            ),
            q_0=q_0
        ),
        apply_update=lambda agent, action, reward: agent.update(
            action=action,
            reward=reward,
            lr=0.1
        )
    ),
    ParameterStudy(
        "Gradient bandit ", [1/8, 1/4, 1/2, 1, 2, 2.5, 3, 3.5],
        apply_parameter=lambda lr: GradientBanditAgent(
            K=K,
            use_baseline=True,
            lr=lr
        ),
        apply_update=lambda agent, action, reward: agent.update(
            action=action,
            reward=reward
        )
    )
]

for parameter in parameter_agent:
    while not parameter.is_done():
        avg = RunningAverage()
        for _ in range(EVALUATION_RUNS):
            rewards = []
            agent = parameter.get_agent()
            bandit = StaticBandit(K)
            for _ in range(EPOCH_LENGTH):
                action = agent.action()
                reward = bandit(action)

                parameter.apply_update(
                    agent=agent,
                    action=action, 
                    reward=reward,
                )
                rewards.append(reward)
            avg.update(sum(rewards) / len(rewards))
        parameter.add_results(avg.value)
        parameter.next()

for results in parameter_agent:
    plt.plot(results.x, results.y, label=results.name)

plt.legend(loc="lower right")
plt.xlabel("Parameter value")
plt.ylabel("Average reward")
plt.savefig('parameter_study.png')
