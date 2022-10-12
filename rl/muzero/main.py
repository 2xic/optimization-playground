from optimization_utils.utils.StopTimer import StopTimer
from optimization_utils.replay_buffers.ReplayBuffer import ReplayBuffer
from model import Model
import torch
from optimization_utils.envs.SimpleEnv import SimpleEnv
from optimization_utils.tree.MonteCarloTreeSearch import MonteCarloTreeSearch
from optimization_utils.tree.MuzeroSimpleEnvState import MuzeroSimpleEnvState
from optimization_utils.tree.SimpleEnvState import SimpleEnvState
from optimization_utils.tree.MuzeroMonteCarloNode import MuzeroMonteCarloNode
from parameters import ROLLOUT_STEPS_K
from optimization_utils.exploration.EpsilonGreedy import EpsilonGreedy
from optimization_utils.diagnostics.Diagnostics import Diagnostics

def learn(replay_buffer: ReplayBuffer, model: Model, diagnostics: Diagnostics, optimizer: torch.optim.Adam):
    trajectory = replay_buffer.sample_trajectory()
    observation = trajectory[0].state

    representation = model.get_state_representation(observation.reshape((1, -1)).float())

    
    loss = torch.tensor(0).float()
    for index in range(ROLLOUT_STEPS_K):
        trajectory_idx = trajectory[index]
        real_action = trajectory_idx.action
        real_reward = trajectory_idx.reward

        policy = torch.tensor(trajectory_idx.metadata['policy'])
        value = torch.tensor(trajectory_idx.metadata['value'])

        predicted_reward, next_state = model.get_dynamics(representation, torch.tensor([real_action]).reshape((1, -1)).float())
        predicted_policy, predicted_value = model.get_prediction(next_state)

        loss_function = lambda x, y, train=True: ((x - y) ** 2).sum() if train else torch.tensor(0)

        reward_loss = loss_function(
            real_reward,
            predicted_reward
        )
        policy_loss = loss_function(
            # I think something is wrong when training the policy
            # Try to train without it and see what happens.
            policy,
            predicted_policy,
            train = False
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
        
def play(replay_buffer: ReplayBuffer, exploration: EpsilonGreedy, model: Model, diagnostics: Diagnostics, epoch_id:int):
    env = SimpleEnv().reset()
    state = env.state
    sum_reward = 0
    actions = []

    while not env.done():
        """
        TODO: 
            Dynamics and the reward components should be added here.
            Need to make some adjustments to the monte carlo logic to make sure it takes
            into account the new information about the reward.
        """
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
        # do the search
        planner.get_action(iterations=100)
        # we use muzero

        if action is None:
            action = planner.root.muzero_action

        if epoch_id % 100 == 0:
            print(env.state)
            print(action)
            print(planner.muzero_policy)

        diagnostics.profile(action)

        state, reward, _, _ = env.step(action)


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

while not timer.is_done():
    reward, actions = play(replay_buffer, exploration, model, diagnostics, timer.epoch)
    loss = learn(replay_buffer, model, diagnostics, optimizer)

    if timer.epoch % 10 == 0:
        diagnostics.print(timer.epoch, {
            "loss": loss,
            "epsilon": exploration.epsilon,
        })

    timer.tick()

