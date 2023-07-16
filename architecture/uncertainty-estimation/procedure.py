from optimization_playground_shared.models.BasicConvModel import BasicConvModel
import torch
import torch.optim as optim

import torch
from torch.autograd import Variable
from tqdm import tqdm

from optimization_playground_shared.dataloaders.Mnist import get_dataloader
import torch
import random

def generate_adversarial(model, X: torch.Tensor, y: torch.Tensor, eps=.007):
    model.zero_grad()
    X_single = Variable(X, requires_grad=True)

    y_pred = model(X_single)
    loss = torch.nn.CrossEntropyLoss()(y_pred, y)
    loss.retain_grad()
    loss.backward()

    fast_gradient = eps * torch.sign(X_single.grad)

    model.zero_grad()
    return X + fast_gradient

class Models:
    def __init__(self, M):
        self.models = [
            BasicConvModel()
            for _ in range(M)
        ]
        self.optimizers = [
            optim.Adam(i.parameters())
            for i in self.models
        ]
        self.index = 0

    def get_random_model(self):
        return self.models[random.randint(0, len(self.models) - 1)]

    def iter(self):
        return iter(zip(self.models, self.optimizers))
    
    def predict(self, X):
        output = self.models[0](X)
        for i in range(1, len(self.models)):
            output += self.models[i](X)
        return output / len(self.models)

    def __len__(self):
        return len(self.models)
    
class Ensemble:
    def __init__(self, M=5) -> None:
        self.models = Models(M)
        self.loss = torch.nn.CrossEntropyLoss()


    def fit(self, X, y):
        losses = []
        for (model, optimizer) in self.models.iter():
            adversarial_x = generate_adversarial(
                model,
                X,
                y
            )
            y_predicted_adversarial = model(adversarial_x)
            y_predicted = model(X)
            loss = self.loss(
                y_predicted_adversarial,
                y
            ) + self.loss(
                y_predicted,
                y
            )
            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(torch.tensor([loss.item()]))
        return torch.concat(losses).mean()

    def predict(self, X):
        return self.models.predict(X)
    
    def eval(self, dataloader):
        accuracy = torch.tensor(0.0)
        length = 0
        for (X, y) in dataloader:
            X = generate_adversarial(
                self.models.get_random_model(),
                X,
                y
            )            
            y_pred = self.predict(X)
            accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += X.shape[0]
        accuracy = (accuracy / length) * 100 
        return accuracy

    def __len__(self):
        return len(self.models)
    
if __name__ == "__main__":
    (train, test) = get_dataloader()

    models = Ensemble(M=1)
    print(f"Training {len(models)} essmeble models")
    progress = tqdm(train)
    for (X, y) in progress:
        loss = models.fit(X, y)
        progress.set_description(f"loss {loss}")
    accuracy = models.eval(test)
    print(f"accuracy: {accuracy}%")

    print("=" * 32)

    models = Ensemble()
    print(f"Training {len(models)} essmeble models")
    progress = tqdm(train)
    for (X, y) in progress:
        loss = models.fit(X, y)
        progress.set_description(f"loss {loss}")
    accuracy = models.eval(test)
    print(f"accuracy: {accuracy}%")

