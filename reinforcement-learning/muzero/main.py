from optimization_utils.utils.StopTimer import StopTimer
from optimization_utils.replay_buffers.ReplayBuffer import ReplayBuffer
from model import Model
import torch
from optimization_utils.envs.SimpleEnv import SimpleEnv
from optimization_utils.tree.MonteCarloTreeSearch import MonteCarloTreeSearch
from optimization_utils.tree.MuzeroSimpleEnvState import MuzeroSimpleEnvState
from optimization_utils.tree.MuzeroMonteCarloNode import MuzeroMonteCarloNode
from parameters import ROLLOUT_STEPS_K
from optimization_utils.exploration.EpsilonGreedy import EpsilonGreedy
from optimization_utils.diagnostics.Diagnostics import Diagnostics
import torch
import numpy as np
from optimization_playground_shared.plot.Plot import Plot, Figure
import random

class RandomAgent:
    def __init__(self, env):
        self.action_space = 2
        self.env = env

    def play(self):
        self.env.reset()
        terminated = False
        sum_reward = 0
        while not terminated:
            action = random.randint(0, self.action_space - 1)
            (
                _,
                reward,
                terminated,
                _
            ) = self.env.step(action)
            sum_reward += reward
        return sum_reward


def learn(replay_buffer: ReplayBuffer, model: Model, diagnostics: Diagnostics, optimizer: torch.optim.Adam):
    trajectory = replay_buffer.sample_trajectory()

    loss = torch.tensor(0).float()

    diagnostics.isValueChanging("trajectory_id", trajectory[0].id)

    for i in range(len(trajectory) - ROLLOUT_STEPS_K):
        observation = trajectory[i].state
        representation = model.get_state_representation(
            observation.reshape((1, -1)).float())

        for index in range(ROLLOUT_STEPS_K):
            trajectory_idx = trajectory[i + index]
            real_action = trajectory_idx.action
            real_reward = trajectory_idx.reward

            policy = torch.tensor(trajectory_idx.metadata['policy'])
            value = torch.tensor(trajectory_idx.metadata['value'])

            predicted_reward, next_state = model.get_dynamics(
                representation, torch.tensor([real_action]).reshape((1, -1)).float())
            predicted_policy, predicted_value = model.get_prediction(
                next_state)

            def loss_function(x, y, train=True): return (
                (x - y) ** 2).sum() if train else torch.tensor(0)

            reward_loss = loss_function(
                real_reward,
                predicted_reward
            )
            policy_loss = loss_function(
                # I think something is wrong when training the policy
                # Try to train without it and see what happens.
                policy,
                predicted_policy,
                train=True
            )
            value_loss = loss_function(
                value,
                predicted_value
            )
            loss += reward_loss + policy_loss + value_loss

            """
            TODO: idea, track each part of the loss, if it goes out of the mean, report it as a probable error in the env.
                    or something else.
            
            diagnostics.loss({
                "reward":  reward_loss,
                "policy": policy_loss,
                "value": value_loss
            })
            """

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def play(replay_buffer: ReplayBuffer, exploration: EpsilonGreedy, model: Model, diagnostics: Diagnostics, epoch_id: int):
    env = SimpleEnv().reset()
    state = env.state
    sum_reward = 0
    actions = []

    should_log = epoch_id % 100 == 0

    while not env.done():
        action = exploration.get_action(lambda: None)

        planner = MonteCarloTreeSearch(
            MuzeroSimpleEnvState(
                model.get_state_representation(
                    env.state.reshape((1, -1)).float()
                ),
                legal_actions=env.legal_actions,
                action_space=env.action_space,
                dynamics=model.get_dynamics,
                predictor=model.get_prediction
            ),
            node=MuzeroMonteCarloNode
        )
        # add noise to the search.
        planner.root._policy = torch.tensor(
            np.random.dirichlet((1, planner.root.state.action_space,)))

        # do the search
        planner.get_action(iterations=100)
        if action is None:
            # TODO: This should instead be the softmax alpha
            action = np.random.choice(
                list(range(planner.root.state.action_space)),
                p=planner.root.policy
            )

        state_before_action = env.state.clone()

        diagnostics.profile(action)

        state, reward, _, _ = env.step(action)

        if should_log:
            print(f"state: ({env.index}) : {state_before_action}")
            print(f"action : {action}")
            print(f"reward : {reward}")
            print(f"policy: {planner.muzero_policy}")
            print("")

        replay_buffer.push(
            state=state,
            reward=reward,
            action=action,
            metadata={
                "policy": planner.muzero_policy,
                "value": planner.muzero_value
            },
            id=epoch_id
        )
        sum_reward += reward
        actions.append(action)

    if should_log:
        print(f"Sum reward {sum_reward}")
        print("")

#    assert diagnostics.is_healthy()
    diagnostics.reward(sum_reward)

    return sum_reward, actions


timer = StopTimer(
    iterations=10_000
)
replay_buffer = ReplayBuffer()
model = Model(
    state_size=2,
    action_space=2
)
optimizer = torch.optim.Adam(model.parameters())
exploration = EpsilonGreedy(
    actions=1
)

timer.tick()

diagnostics = Diagnostics()

muzero_rewards = []
optimal_reward = []
random_agent_epochs = []

sum_append = lambda x, y: (x[-1] if len(x) > 0 else 0) + y 

while not timer.is_done():
    reward, actions = play(replay_buffer, exploration,
                           model, diagnostics, timer.epoch)
    loss = learn(replay_buffer, model, diagnostics, optimizer)

    if timer.epoch % 10 == 0:
        diagnostics.print(timer.epoch, {
            "loss": loss,
            "epsilon": exploration.epsilon,
            "last_game_sum_reward": reward,
        })
    
    random_agent = RandomAgent(SimpleEnv())

    random_agent_epochs.append(sum_append(random_agent_epochs, random_agent.play().item()))
    muzero_rewards.append(sum_append(muzero_rewards, reward.item()))
    optimal_reward.append(sum_append(optimal_reward, 10))

    timer.tick()
    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "random agent": random_agent_epochs,
                    "optimal": optimal_reward,
                    "muzero": muzero_rewards,
                },
                title="Agent vs random agent",
                x_axes_text="Timestamp",
                y_axes_text="Sum reward over time",
            ),
        ],
        name='evaluation.png'
    )
