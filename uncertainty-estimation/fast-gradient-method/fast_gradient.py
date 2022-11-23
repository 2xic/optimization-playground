import torch
from torch.autograd import Variable

def fast_gradient(model, X, y, eps=.007):
    model.zero_grad()
    X = X.reshape((1, ) + X.shape)
    X = Variable(X, requires_grad=True)

    y = torch.tensor([y])
    y = y.reshape((1))

    y_pred = model(X)
#    loss = torch.nn.NLLLoss()(y_pred, y)
    loss = torch.nn.CrossEntropyLoss()(y_pred, y)
    loss.retain_grad()
    loss.backward()

    fast_gradient = eps * torch.sign(X.grad)
    return fast_gradient
