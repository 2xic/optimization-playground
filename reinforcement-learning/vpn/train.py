from statistics import mode
from simple_env_trajectory import play
from torch_model import TorchModel
from torch.optim import Adam
from epsilon import Epsilon
from optimization_playground_shared.plot.Plot import Plot, Figure
import random
from optimization_utils.envs.SimpleEnv import SimpleEnv

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

model = TorchModel()
optimizer = Adam(
    model.model.parameters(),
    lr=3e-4
)
epsilon = Epsilon()


vpn_rewards = []
optimal_reward = []
random_agent_epochs = []
sum_append = lambda x, y: (x[-1] if len(x) > 0 else 0) + y 

for epoch in range(1_000):
    (loss, _, total_reward) = play(model, epsilon)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(loss, total_reward, epsilon.epsilon)
    
    random_agent = RandomAgent(SimpleEnv())

    random_agent_epochs.append(sum_append(random_agent_epochs, random_agent.play().item()))
    vpn_rewards.append(sum_append(vpn_rewards, total_reward.item()))
    optimal_reward.append(sum_append(optimal_reward, 10))

    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "random agent": random_agent_epochs,
                    "optimal": optimal_reward,
                    "vpn": vpn_rewards,
                },
                title="Agent vs random agent",
                x_axes_text="Timestamp",
                y_axes_text="Sum reward over time",
            ),
        ],
        name='evaluation.png'
    )
