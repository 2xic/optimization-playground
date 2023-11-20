"""
Hm - this doesn't seem to converge at all.

"""
import torch
import torch.nn as nn
import random

def sigmoid_deriv(x):
    return x * (1 - x)

def generate_x_y_dataset(batch_size):
    x = []
    y = []
    size = 3
    for i in range(batch_size):
        a = []
        b = []
        for i in range(size):
            value_a = random.randint(0, 1)
            value_b = 1 - value_a
            a.append(value_a)
            b.append(value_b)
        x.append(a)
        y.append(b)
    return x, y

#x, y = generate_x_y_dataset(4)
X = torch.tensor([
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1]
]).float()
y = torch.tensor([
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 0]
]).float()
layer_1 = nn.Linear(3, 4).requires_grad_(False)
layer_2 = nn.Linear(4, 3).requires_grad_(False)

# predicts synthetic gradients
synthetic_layer_2 = nn.Sequential(*(    
    nn.BatchNorm1d(3),
    nn.Linear(3, 4),
    nn.Dropout(p=0.8),
    nn.ELU(),
    nn.Linear(4, 3),
    nn.Tanh(),
))

synthetic_optimizer = torch.optim.Adam(
    list(synthetic_layer_2.parameters()), 
    lr=0.001
)

lr = 1 #!  0.001
train_with_synthetics = True
if train_with_synthetics: 
    lr = 0.001
for j in range(100_000):
    # forward pass
    l1 = torch.sigmoid(layer_1(X))
    l2 = torch.sigmoid(layer_2(l1))

  #  predicted_gradient_1 = synthetic_layer_1(l1)
    #with torch.no_grad():
    if True:
        predicted_gradient_2 = synthetic_layer_2(l2)

        # calculate the gradient error
        l2_grad = (y - l2)  * sigmoid_deriv(l2) 
        # This error can be predicted
        # predict the above ^ 
        synthetic_gradient = predicted_gradient_2.clone().detach() * 0.001 #* torch.rand(predicted_gradient_2.shape)
        copy_synthetic_gradient = l2_grad if not train_with_synthetics else synthetic_gradient
        # I think I need to update the model with this gradient also ? 
        l1_grad = copy_synthetic_gradient.matmul(layer_2.weight) * sigmoid_deriv(l1)
        # ^ then you get this ? 

        # update the error broski
 #       layer_2.weight -= l1.T.mm(copy_synthetic_gradient).T * lr 
#        layer_1.weight -= X.T.mm(l1_grad).T * lr 
        layer_2.weight += l1.T.mm(copy_synthetic_gradient).T * lr 
        layer_1.weight += X.T.mm(l1_grad).T * lr 

    if j % 1_000 == 0:
        print("=" * 4 + "\t" + str(j) + "\t" + "=" * 4)
        print("predicted")
        print(l2.round())
        print("truth")
        print(y)
        print("")
        print("l2 raw output")
        print(layer_2(l1))
        print("grads ")
        print((l2_grad))
        print((predicted_gradient_2))
        print((
            synthetic_layer_2(l2) - 
            l2_grad
        ).mean() )
        print("acc ", (torch.abs(y - l2) < 0.01).sum())
        print("")

    if train_with_synthetics:
        synthetic_gradient_error = (
            synthetic_layer_2(l2) - 
            l2_grad
        ).mean() 

        synthetic_optimizer.zero_grad()
        synthetic_gradient_error.backward()
        synthetic_optimizer.step()
        
