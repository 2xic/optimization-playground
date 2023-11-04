from model import SimpleQLearning
from env import GamePool
import multiprocessing
import torch.nn as nn
import torch.optim as optimi

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    model = SimpleQLearning(
        state_size=180,
        action_size=5
    )
    optimizer = optimi.Adam(model.parameters())
    game_pool = GamePool(n=15)
    for epoch in range(100):
        runs = game_pool.run(model)
        print("Runs == ", len(runs))
        loss = 0
        total_score = 0
        for i in runs:
            for (prev_state, state, action, returns) in i.get_state_action_pairs():
                model_output = model(prev_state)
                true_output = model_output.clone()
                true_output[0][action] = returns
                error = nn.MSELoss()(model_output, true_output)
                loss += error
            total_score += i.score
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {epoch}, Loss: {loss}, Score {score}".format(epoch=epoch, loss=loss, score=total_score))
