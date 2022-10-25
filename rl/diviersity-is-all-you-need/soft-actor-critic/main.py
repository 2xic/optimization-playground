import torch
from optimization_utils.envs.SimpleEnv import SimpleEnv
from optimization_utils.exploration.EpsilonGreedyManual import EpsilonGreedyManual
import random
from optimization_utils.diagnostics.Diagnostics import Diagnostics
from model import SAC
from optimization_utils.replay_buffers.ReplayBuffer import ReplayBuffer

env = SimpleEnv()
replay_buffer = ReplayBuffer()

def play(model: SAC, diagnostics: Diagnostics):
    global env, replay_buffer
    env.reset()

    while not env.done():
        action_distribution = model(env.env.float())
        #print(action_distribution)
        action = epsilon.get_action(lambda: torch.argmax(action_distribution).item())

        (_, reward, action, _) = env.step(action)
        total_reward += reward

        game.add(action_distribution, action)
        diagnostics.profile(action)
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
    exit(0)

if __name__ == "__main__":
    model = SAC(
        input_size=env.state_size,
        output_size=env.action_space,
    )
    diagnostics = Diagnostics()

    for epoch in range(10_000):
        loss = play(model, diagnostics)

        if epoch % 100 == 0:
            diagnostics.print(epoch, metadata={
                "loss": loss.item() 
            })

