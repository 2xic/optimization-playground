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

def learn(replay_buffer: ReplayBuffer, model: Model, optimizer: torch.optim.Adam):
    #                           <- maybe replay buffer should give trajectory ids :/
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

        loss_function = lambda x, y: ((x - y) ** 2).sum()

        loss += loss_function(
            real_reward,
            predicted_reward
        ) + loss_function(
            policy,
            predicted_policy
        ) + loss_function(
            value,
            predicted_value
        )

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
        
def play(replay_buffer: ReplayBuffer, model: Model, epoch_id:int):
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
        planner = MonteCarloTreeSearch(
            MuzeroSimpleEnvState(
                model.get_state_representation(
                    torch.tensor(env.state).reshape((1, -1)).float()
                ),
                legal_actions=env.legal_actions,
                action_space=env.action_space,
                dynamics=model.get_dynamics
            ),
            node=MuzeroMonteCarloNode
        ) 
        # do the search
        planner.get_action(iterations=100)
        # we use muzero
        action = planner.root.muzero_action

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

timer.tick()

while not timer.is_done():
    reward, actions = play(replay_buffer, model, timer.epoch)
    loss = learn(replay_buffer, model, optimizer)

    if timer.epoch % 10 == 0:
        print(timer.epoch, reward, actions, loss)

    timer.tick()

