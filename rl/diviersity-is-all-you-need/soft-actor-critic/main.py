from functools import total_ordering
import torch
from optimization_utils.envs.SimpleEnv import SimpleEnv
from optimization_utils.exploration.EpsilonGreedyManual import EpsilonGreedyManual
import random
from optimization_utils.diagnostics.Diagnostics import Diagnostics
from model import SAC
from optimization_utils.replay_buffers.ReplayBuffer import ReplayBuffer
import gym

env =  gym.make('CartPole-v0') #SimpleEnv()
replay_buffer = ReplayBuffer()

def play(model: SAC, diagnostics: Diagnostics):
    global env, replay_buffer
    env.reset() 

    total_reward = 0
    done = False
    while not done: # env.done():
        prev_state = torch.tensor(list(env.state)).clone() # torch.from_numpy(env.state)
        action, _ = model.action(prev_state)
        action = action.item()
#        print(action)
     #   print(env.step(action))
        (_, reward, done, _, _) = env.step(action)
        total_reward += reward

    #    print(env.state)

        diagnostics.profile(int(action))
        diagnostics.reward(reward)
        replay_buffer.push(
            state=prev_state,
            reward=reward,
            action=action,
            is_last_state=done, #env.done(),
            metadata={
                "next_state": torch.tensor(list(env.state)).clone(),
            },
            id=-1
        )
    return total_reward

if __name__ == "__main__":
    model = SAC(
        input_size=4, #env.state_size,
        output_size=2, #env.action_space,
    )
    diagnostics = Diagnostics()

    for epoch in range(10_000):
        total_reward = play(model, diagnostics)

        if epoch % 100 == 0 and epoch > 0:
            #for i in range(100):
            metadata = model.optimize(replay_buffer)
            metadata['total_reward'] = total_reward

            diagnostics.print(epoch, metadata=metadata)
            """
            metadata = {}
            metadata["policy_view_zero_state"] = str(model.action(torch.tensor([[1, 0]]).float())[1].data)
            metadata["policy_view_one_state"] = str(model.action(torch.tensor([[0, 1]]).float())[1].data)

            metadata["q_view_zero_state"] = str(model.q_1(torch.tensor([[1, 0]]).float()).data)
            metadata["q_view_one_state"] = str(model.q_2(torch.tensor([[0, 1]]).float()).data)
            diagnostics.model_print(metadata=metadata)
            """
