"""
Hm - it kinda works, but not optimally
"""
import torch
import torch.nn as nn

def sigmoid_deriv(x):
    return x * (1 - x)

def train(
    synthetic_optimizer_lr= 0.001,
    train_with_synthetics = True,
    log=True
):
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
    layer_1 = nn.Linear(3, 4).requires_grad_(False)
    layer_2 = nn.Linear(4, 1).requires_grad_(False)

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
    alpha = 0.1
    for j in range(100_000):
        # forward pass
        l1 = torch.sigmoid(layer_1(X))
        l2 = torch.sigmoid(layer_2(l1))

        if True:
            predicted_gradient_2 = synthetic_layer_2(l1).detach()

            # calculate the gradient error
            l2_grad = (y - l2)  * sigmoid_deriv(l2) 
            # This error can be predicted
            # I think I need to update the model with this gradient also ? 
            synthetic_gradient = -predicted_gradient_2.clone().detach() * 1e-4
            l1_grad = synthetic_gradient.matmul(layer_2.weight) * sigmoid_deriv(l1)

            # update the error
            layer_2.weight += alpha * l1.T.mm(l2_grad).T
            layer_1.weight += alpha * X.T.mm(l1_grad).T

            for i in [layer_2, layer_1]:
                i.weight[i.weight < -10] = -10
                i.weight[i.weight > 10] = 10

        synthetic_gradient_error = None
        if train_with_synthetics:
            # need to help the model some ? 
            for i in range(5):
                synthetic_gradient_error = nn.functional.mse_loss(
                    synthetic_layer_2(l1),
                    l2_grad
                )
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

if __name__ == "__main__":
    train(
        synthetic_optimizer_lr=1e-3,
        train_with_synthetics=True
    )
