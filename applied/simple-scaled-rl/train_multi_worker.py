from model import SimpleQLearning
from env import GamePool
import multiprocessing
import torch.nn as nn
from model_worker_pool import ModelPool
from optimization_playground_shared.plot.Plot import Plot, Figure

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    model = SimpleQLearning(
        state_size=180,
        action_size=5
    )
    model_pool = ModelPool(
        n=5,
        contract_sub_model=lambda: SimpleQLearning(
            state_size=180,
            action_size=5
        )
    )
    game_pool = GamePool(n=15)
    scores_over_time = []
    for epoch in range(100):
        runs = game_pool.run(model)
        print("Runs == ", len(runs))
        loss = 0
        total_score = 0
        for i in runs:
            model_specific_parameters = model_pool[i.model_id]
            for (prev_state, state, action, returns) in i.get_state_action_pairs():
                model_output = model_specific_parameters(prev_state)
                true_output = model_output.clone()
                true_output[0][action] = returns
                # backward the gradients on this level only
                error = nn.functional.mse_loss(model_output, true_output)
                error.backward()
                loss += error
            total_score += i.score
        #model.zero_grad()
        #loss.backward()
        #optimizer.step()
        model_pool.step_model()

        print("Epoch {epoch}, Loss: {loss}, Score {score}".format(epoch=epoch, loss=loss, score=total_score))
        scores_over_time.append(total_score)
        # plot the results
        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Score": scores_over_time,
                    },
                    title="Scores over time",
                    x_axes_text="Epochs",
                    y_axes_text="Score",
                ),
            ],
            name=f'plots/multiple_model_trained.png'
        )
