from env.simple_rl_env import SimpleRlEnv
from config import Config
from agent import Agent
from debug import Debug

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
env = SimpleRlEnv()
agent = Agent(config, env)
debug = Debug()

for epoch in range(1_000):
    # After a certain amount of epochs let's switch to using the models policy
    if epoch > 10:
        config.is_training = False

    sum_reward = agent.play(
        debugger=debug,
    )
    loss = agent.loss()

    debug.add(
        loss,
        sum_reward
    )
    debug.print()
