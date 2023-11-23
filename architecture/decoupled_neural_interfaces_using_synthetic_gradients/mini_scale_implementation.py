"""
Hm - it kinda works, but not optimally
"""
import torch
import torch.nn as nn
from optimization_playground_shared.plot.Plot import Plot, Figure

X = torch.tensor([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
]).float()

y = torch.tensor([
    [0],
    [0],
    [1],
    [1]
]).float()

def sigmoid_deriv(x):
    return x * (1 - x)

def get_synthetic_model(
    synthetic_optimizer_lr
):
    # predicts synthetic gradients
    layer = nn.Linear(4, 1,)
    stdv = 0.2
    layer.weight.data.uniform_(-stdv, stdv)

    synthetic_layer_2 = nn.Sequential(*(    
        nn.BatchNorm1d(4),
        nn.Dropout(p=0.2),
        layer,
        nn.ELU()
    ))

    synthetic_optimizer = torch.optim.Adam(
        synthetic_layer_2.parameters(), 
        lr=synthetic_optimizer_lr
    )

    return synthetic_layer_2, synthetic_optimizer

def train(
    synthetic_optimizer_lr = 0.001,
    train_synthetics_model = True,
    log=True,
    times_to_retrain_on_error=5,
    keep_model_in_scope=False,
    epochs=10_000,
):
    layer_1 = nn.Linear(3, 4).requires_grad_(False)
    layer_2 = nn.Linear(4, 1).requires_grad_(False)

    synthetic_layer_2, synthetic_optimizer = get_synthetic_model(
        synthetic_optimizer_lr=synthetic_optimizer_lr,
    )

    alpha = 0.1
    error_over_time = []
    for j in range(epochs):
        # forward pass
        l1 = torch.sigmoid(layer_1(X))
        l2 = torch.sigmoid(layer_2(l1))

        predicted_gradient_2 = synthetic_layer_2(l1).detach()

        # calculate the gradient error
        l2_grad = (y - l2)  * sigmoid_deriv(l2) 
        synthetic_gradient = torch.rand(4, 1)
        if train_synthetics_model:
            synthetic_gradient = predicted_gradient_2.clone().detach() # predicts the l2 grad
        l1_grad = synthetic_gradient.matmul(layer_2.weight) * sigmoid_deriv(l1)

        # update the error
        layer_2.weight += alpha * l1.T.mm(l2_grad).T
        layer_1.weight += alpha * X.T.mm(l1_grad).T

        if keep_model_in_scope:
            for i in [layer_2, layer_1]:
                i.weight[i.weight < -10] = -10
                i.weight[i.weight > 10] = 10

        synthetic_gradient_error = None
        if train_synthetics_model:
            # need to help the model some ? 
            for i in range(times_to_retrain_on_error):
                synthetic_gradient_error = nn.functional.mse_loss(
                    synthetic_layer_2(l1),
                    l2_grad
                )
                synthetic_optimizer.zero_grad()
                # Using MAE won't work at all
                #synthetic_gradient_error = (
                #    synthetic_layer_2(l1) - 
                #    l2_grad
                #).mean()
                synthetic_optimizer.zero_grad()
                synthetic_gradient_error.backward()
                synthetic_optimizer.step()


        if j % 1_000 == 0 and log:
            print("=" * 4 + "\t" + str(j) + "\t" + "=" * 4)
            print("predicted")
            print(l2)
            print("error")
            print(synthetic_gradient_error,)
            print("truth")
            print(y)
            print("")
            print("l2 raw output")
            print(layer_2(l1))
            print("grads ")
            print(("real", l2_grad))
            print(("predicted", predicted_gradient_2))
            print()
            print("acc ", (torch.abs(y - l2) < 0.01).sum())
            print("")
        error_over_time.append(((l2 - y) ** 2).sum(dim=0).item())
    return error_over_time

def plot(
    error_synthetics_gradient,
    error_no_training_model,
):
    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "error NN prediction gradient": error_synthetics_gradient,
                    "error rand gradient": error_no_training_model,
                },
                title="Variants of synectics gradients",
                x_axes_text="Epochs",
                y_axes_text="MSE",
            ),
        ],
        name='example.png'
    )

if __name__ == "__main__":
    error_synthetics_gradient = train(
        synthetic_optimizer_lr=1e-3,
        train_synthetics_model=True,
        times_to_retrain_on_error=1,
        epochs=1_000,
    )
    error_no_training_model = train(
        synthetic_optimizer_lr=1e-3,
        # don't train the model ... expects to not learn anything 
        train_synthetics_model=False,
        times_to_retrain_on_error=1,
        epochs=1_000,
    )
    plot(
        error_synthetics_gradient=error_synthetics_gradient,
        error_no_training_model=error_no_training_model,
    )
