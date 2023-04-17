from epsilon import Epsilon
from model import DqnModel
from optimization_utils.envs.TicTacToe import TicTacToe
import torch
from optimization_utils.plotters.SimplePlot import SimplePlot
from optimization_utils.plotters.LinePlot import LinePlot

def play(env, model: DqnModel, epsilon: Epsilon, optimizer: torch.optim.Adam):
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
       #     if action not in env.legal_actions:
       #         print(f"Illegal tried to be preformed, action form model: {is_model_action}")
       #         print(model_output)
       #         break
        #   print(f"Model: {is_model_action} Action: {action}, Legal: {env.legal_actions}")
            previous_state = env.env.clone().float()

            (state, reward, action, gamma) = env.step(action)
            iterations.append([previous_state, action, reward])
            total_reward += reward
        env.reset()
    #if random.randint(0, 10) == 0:
    #    print(total_reward)

    for (state, action, reward) in iterations:
        predicted_state = model(state).detach()
        #         target_reward = reward + gamma * model(state)[0][torch.argmax(model(state)[0])].item()
        #       ^ I think this is the more "correct" reward, but this one scales nicely also
        predicted_state[0][action] = reward
        prediction = model(state)

        loss = torch.nn.MSELoss()(predicted_state, prediction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_reward

if __name__ == "__main__":
    env = TicTacToe()
    epsilon = Epsilon()
    model = DqnModel(env.action_space)
    optimizer = torch.optim.Adam(model.parameters())
    rewards = []
    for i in range(5_00):
        reward = play(env, model, epsilon, optimizer)
        if i % 10 == 0:
            print(f"Epoch: {i}, Reward: {reward}, Epsilon {epsilon.epsilon}")
        epsilon.update()
        rewards.append(reward)

    training_accuracy = SimplePlot()
    training_accuracy.plot(
        [
            LinePlot(y=rewards, legend="Q-learning", title="Reward for tic tac toe", x_text="Iteration", y_text="Reward over time", y_min=-10, y_max=10),
        ]
    )
    training_accuracy.save("rewards.png")
