from env.simple_rl_env import SimpleRlEnv
from config import Config
from agent import Agent
from debug import Debug
from random_agent import RandomAgent
from optimization_playground_shared.plot.Plot import Plot, Figure
from average_over_time import EvaluationOverTime

def train_one():
    config = Config(
        is_training=True,
        num_actions=2,
        state_size=2,
        max_iterations=1_00,
        max_depth=5,
        # model config
        projection_network_output_size=16,
        state_representation_size=4,
        lr=0.00001,
        # numbers appendix 3
        c_1=1.25, 
        c_2=19652,
    )
    max_epochs = 1_00

    agent = Agent(config, SimpleRlEnv())
    random_agent = RandomAgent(config, SimpleRlEnv())
    debug = Debug()

    # Results
    random_agent_scores = []
    agent_scores = []
    sum_loss_over_time = []

    for epoch in range(max_epochs):
        # After a certain amount of epochs let's switch to using the models policy
        if epoch > 10:
            config.is_training = False

        sum_reward = agent.play(
            debugger=debug,
        )
        sum_reward_random_agent = random_agent.play()
        loss = agent.loss()

        debug.add(
            loss,
            sum_reward
        )
        debug.print()
        
        get_prev_element = lambda arr : (0 if len(arr) == 0 else arr[-1])
        agent_scores.append(sum_reward + get_prev_element(agent_scores))
        random_agent_scores.append(sum_reward_random_agent + get_prev_element(random_agent_scores))
        sum_loss_over_time.append(loss)

    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "sum_loss_over_time": sum_loss_over_time,
                },
                title="Agent vs random agent",
                x_axes_text="Timestamp",
                y_axes_text="Sum reward",
            ),
            Figure(
                plots={
                    "loss_policy": agent.loss_debug.loss_policy,
                    "loss_reward": agent.loss_debug.loss_reward,
                },
                title="Loss",
                x_axes_text="Timestamp",
                y_axes_text="Sum reward",
            ),
        ],
        name='loss.png'
    )

    #plot = Plot()
    #plot.plot_figures(
    #    figures=[
    #        Figure(
    #            plots={
    #                "random_agent": random_agent_scores,
    #                "agent": agent_scores,
    #            },
    #            title="Agent vs random agent",
    #            x_axes_text="Timestamp",
    #            y_axes_text="Sum reward",
    #        ),
    #    ],
    #    name='evaluation.png'
    #)
    return (
        random_agent_scores,
        agent_scores,
    )

if __name__ == "__main__":
    eval_agent = EvaluationOverTime()
    eval_random_agent = EvaluationOverTime()

    sum_reward_agent = []
    sum_reward_random_agent = []
    for _ in range(10):
        (random_agent_scores, agent_scores) = train_one()
        eval_agent.add(agent_scores)
        eval_random_agent.add(random_agent_scores)

        sum_reward_random_agent.append(sum(random_agent_scores))
        sum_reward_agent.append(sum(agent_scores))

        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "random_agent": eval_random_agent.rewards_epochs,
                        "agent": eval_agent.rewards_epochs,
                    },
                    title="Agent vs random agent",
                    x_axes_text="Timestamp",
                    y_axes_text="Sum reward",
                ),
                Figure(
                    plots={
                        "random_agent": sum_reward_random_agent,
                        "agent": sum_reward_agent,
                    },
                    title="Agent vs random agent for training rounds",
                    x_axes_text="Timestamp",
                    y_axes_text="Sum reward",
                ),
            ],
            name='evaluation.png'
        )
