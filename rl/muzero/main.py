from optimization_utils.utils.StopTimer import StopTimer
from optimization_utils.replay_buffers.ReplayBuffer import ReplayBuffer
from model import Model
import torch
from optimization_utils.envs.SimpleEnv import SimpleEnv
from optimization_utils.tree.MonteCarloTreeSearch import MonteCarloTreeSearch
from optimization_utils.tree.SimpleEnvState import SimpleEnvState
from parameters import ROLLOUT_STEPS_K

def learn(replay_buffer: ReplayBuffer, model: Model):
    #                           <- maybe replay buffer should give trajectory ids :/
    trajectory = replay_buffer.sample_trajectory()
    observation = trajectory[0].state
    representation = model.get_state_representation(observation)

    
    loss = torch.tensor(0)
    for index in range(ROLLOUT_STEPS_K):
        trajectory_idx = trajectory[index]
        real_action = trajectory_idx.action
        real_reward = trajectory_idx.reward

        policy = trajectory_idx['metadata'].policy
        value = trajectory_idx['metadata'].value

        predicted_reward, next_state = model.dynamics(representation, real_action)
        predicted_policy, predicted_value = model.prediction(next_state)

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

    # pass
    """
    optimizer.zero_grad()
    loss.backwards()
    optimizer.step()
    """

        
def play(replay_buffer: ReplayBuffer, model: Model):
    env = SimpleEnv()
    planner = MonteCarloTreeSearch(
        SimpleEnvState(env)
    )
    action = planner.get_action()

    state = env.state
    _, reward, _, _ = env.step(action)

    """
        Nooo, the replay buffer should contain the entire game history (trajectory)
    """
    replay_buffer.push(
        state=state,
        reward=reward,
        action=action
    )

    print(action)
    print(reward)

timer = StopTimer()
replay_buffer = ReplayBuffer()
model = Model()

timer.tick()

while not timer.is_done():
    play(replay_buffer, model)
    learn(replay_buffer, model)
    timer.tick()
    break
