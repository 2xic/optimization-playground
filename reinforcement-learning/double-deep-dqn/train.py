from epsilon import Epsilon
from model import DqnModel
from optimization_utils.envs.TicTacToe import TicTacToe
import torch
from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot
import random
import math

def play(env, model: DqnModel, target_model: DqnModel, epsilon: Epsilon, optimizer: torch.optim.Adam):
    iterations = []

    total_reward = 0

    for _ in range(10):
        while not env.done and env.winner is None: # ():
            epsilon_action = epsilon.action_from_array(env.legal_actions)
            model_output = model(env.env.float())
            for i in range(0, env.action_space):
                if i not in env.legal_actions:
                    model_output[0][i] = 0
            (action, is_model_action) = (torch.argmax(model_output), True) if epsilon_action is None else (epsilon_action, False)

            previous_state = env.env.clone().float()

            (state, reward, action, gamma) = env.step(action)
            iterations.append([previous_state, action, reward, state.clone().float(), (env.done or env.winner is not None)])
            total_reward += reward
        env.reset()

    for (prev_state, action, reward, state, _) in iterations:
        predicted_state = model(prev_state).detach()

        target_index = None
        with torch.no_grad():
            target_index = torch.argmax(target_model(state)[0])

        next_reward = model(state)[0][target_index].item()
        target_reward = reward + max(0, next_reward)
        predicted_state[0][action] = target_reward

        prediction = model(state)
    #    print((target_reward, prediction[0][action]))

        loss = torch.nn.MSELoss()(predicted_state, prediction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_reward

if __name__ == "__main__":
    env = TicTacToe()
    epsilon = Epsilon()
    q_1 = DqnModel(env.action_space)
    q_2 = DqnModel(env.action_space)
    
    q1_optimizer = torch.optim.Adam(q_1.parameters())
    q2_optimizer = torch.optim.Adam(q_2.parameters())

    rewards = []
    for i in range(5_00):
        reward = play(
            env, 
            q_1 if random.randint(0, 2) == 1 else q_2, 
            q_2 if random.randint(0, 2) == 1 else q_1, 
            epsilon, 
            q1_optimizer if random.randint(0, 2) == 1 else q2_optimizer, 
        )
        if i % 10 == 0:
            print(f"Epoch: {i}, Reward: {reward}, Epsilon {epsilon.epsilon}")
        epsilon.update()
        rewards.append(reward)

    training_accuracy = SimplePlot()
    training_accuracy.plot(
        [
            LinePlot(y=rewards, legend="Double Q-learning", title="Reward for tic tac toe", x_text="Iteration", y_text="Reward over time", y_min=-10, y_max=10),
        ]
    )
    training_accuracy.save("rewards.png")
