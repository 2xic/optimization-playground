from epsilon import Epsilon
from model import DqnModel
from optimization_utils.envs.SimpleEnv import SimpleEnv
import torch
import random

def play(model: DqnModel, epsilon: Epsilon, optimizer: torch.optim.Adam):
    env = SimpleEnv()

    iterations = []

    total_reward = 0
    while not env.done():
        epsilon_action = epsilon.action()
        action = torch.argmax(model(env.env.float())) if epsilon_action is None else epsilon_action
        previous_state = env.env.float()

        (state, reward, action, gamma) = env.step(action)
        iterations.append([previous_state, action, reward])
        total_reward += reward

    if random.randint(0, 10) == 0:
        print(total_reward)

    for (state, action, reward) in iterations:
        predicted_state = model(state).detach()
        predicted_state[action] = reward
        prediction = model(state)

        loss = torch.nn.MSELoss()(predicted_state, prediction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

if __name__ == "__main__":
    epsilon = Epsilon()
    model = DqnModel()
    optimizer = torch.optim.Adam(model.parameters())

    for _ in range(100):
        play(model, epsilon, optimizer)

