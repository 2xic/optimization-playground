from model import Model
from tinygrad import nn, Tensor
from env.simple_rl_env import SimpleRlEnv
from config import Config
from agent import Agent

config = Config(
    is_training=True,
    num_actions=2,
    state_size=2,
    state_representation_size=4,
    max_iterations=1_00,
    max_depth=5
    # numbers appendix 3
    c_1=1.25, 
    c_2=19652,
)
env = SimpleRlEnv()
agent = Agent(config, env)

for _ in range(100):
    output = agent.play()
    print("one epoch")
#    model.opt.step()

