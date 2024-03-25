from env.simple_rl_env import SimpleRlEnv
from config import Config
from agent import Agent
from debug import Debug
from random_agent import RandomAgent
from optimization_playground_shared.plot.Plot import Plot, Figure
from average_over_time import EvaluationOverTime
import torch
from models.tinygrad_model import Model  as TinygradModel, function_time
from models.torch_model import Model as TorchModel
import argparse
import time
from optimization_playground_shared.utils.Timer import Timer
from optimization_playground_shared.utils.GlobalTimeSpentInFunction import GlobalTimeSpentInFunction

backends = {
    "tinygrad": TinygradModel,
    "torch": TorchModel,
}

parser = argparse.ArgumentParser(
                    prog='efficent zero',
                    description='Test with torch or tinygrad')
parser.add_argument('backend', choices=['tinygrad', 'torch'])
args = parser.parse_args()

def train_one():
    config = Config(
        is_training=True,
        num_actions=2,
        state_size=2,
        max_iterations=3_00,
        max_depth=5,
        replay_buffer_size=100,
        # model config
        projection_network_output_size=16,
        state_representation_size=4,
        lr=13e-4,
        # numbers appendix 3
        c_1=1.25, 
        c_2=19652,
    )
    max_epochs = 1_00

    agent = Agent(config, SimpleRlEnv(), backends[args.backend])
    random_agent = RandomAgent(config, SimpleRlEnv())
    debug = Debug()

    # Results
    random_agent_scores = []
    agent_scores = []
    optimal_reward = []

    sum_loss_over_time = []
    agent_reward_over_time = []
    random_agent_reward_over_time = []

    start = time.time()

    for epoch in range(max_epochs):
        # After a certain amount of epochs let's switch to using the models policy
        # if epoch > 10:
        #config.is_training = random.randint(0, 1) == 0
        config.is_training = epoch % 2 == 1

        with Timer("model_play"):
            _ = agent.play(
                debugger=debug,
            )
        sum_reward = 0
        with torch.no_grad():
            sum_reward = agent.test()
        sum_reward_random_agent = random_agent.play()
        # Should likely iterate over this n times instead of always fetching new data
        loss = None
        with Timer("loss"):
            for _ in range(3):
                loss = agent.loss()

        debug.add(
            loss,
            sum_reward
        )
        debug.print()
        
        get_prev_element = lambda arr : (0 if len(arr) == 0 else arr[-1])
        agent_scores.append(sum_reward + get_prev_element(agent_scores))
        optimal_reward.append(get_prev_element(optimal_reward) + 11)
        random_agent_scores.append(sum_reward_random_agent + get_prev_element(random_agent_scores))
        sum_loss_over_time.append(loss)
        # Reward over time
        agent_reward_over_time.append(sum_reward)
        random_agent_reward_over_time.append(sum_reward_random_agent)

        avg_time = (time.time() - start) / (epoch + 1)

        print("{}: Average time {} epoch".format(args.backend, avg_time))
#        print(GlobalTimeSpentInFunction().timers)
        if avg_time > 10:
            print("")
            print("=============")
            for key, value in sorted(GlobalTimeSpentInFunction().timers.items(), key=lambda x: x[1]):
                print(f"{key}:\t{value}")
            print("====")
            combined = 0
            for key, value in sorted(function_time.items(), key=lambda x: x[1]):
                print(f"{key}:\t{value}")
                combined += value
            print(combined)
            print("Took to long")
            exit(1)

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
                    "loss_projection": agent.loss_debug.loss_projection_loss,
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
        optimal_reward,
        agent_reward_over_time, 
        random_agent_reward_over_time
    )

if __name__ == "__main__":
    eval_agent = EvaluationOverTime()
    eval_random_agent = EvaluationOverTime()

    sum_reward_agent = []
    sum_reward_random_agent = []

    eval_agent_reward_over_time = EvaluationOverTime()
    eval_random_agent_reward_over_time = EvaluationOverTime()

    for _ in range(3):
        (random_agent_scores, agent_scores, optimal_reward, agent_reward_over_time, random_agent_reward_over_time) = train_one(

        )
        eval_agent.add(agent_scores)
        eval_random_agent.add(random_agent_scores)

        sum_reward_random_agent.append(sum(random_agent_scores))
        sum_reward_agent.append(sum(agent_scores))

        eval_agent_reward_over_time.add(agent_reward_over_time)
        eval_random_agent_reward_over_time.add(random_agent_reward_over_time)

        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "random_agent": eval_random_agent.rewards_epochs,
                        "optimal": optimal_reward,
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
                    y_axes_text="Sum total reward",
                ),
                Figure(
                    plots={
                        "agent": eval_agent_reward_over_time.rewards_epochs,
                        "random": eval_random_agent_reward_over_time.rewards_epochs,
                    },
                    title="Reward each epoch",
                    x_axes_text="Timestamp",
                    y_axes_text="Total reward",
                ),
            ],
            name='evaluation.png'
        )
