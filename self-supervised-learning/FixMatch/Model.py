from torch import optim, nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from fixmatch import FixMatch
from Parameters import loss_reduction, output_reduction, supervised_size_ratio


# just using the example model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class SimpleModel:
    def __init__(self, model):
        self.model = model
        self.fix_match_loss = FixMatch()
        self.unlabeled_loss_weight = 1
        self.labeled_loss_weight = 0.6
        self.batch = 0

    def forward(self, X):
        return self.model(X)

    def test_step(self, batch, _):
        x, y = batch

        batch_size = x.shape[0]
        predictions = self.get_class_predictions(x)
        accuracy = (batch_size - torch.count_nonzero(predictions - y)) / float(batch_size)
        self.batch += 1
        return accuracy

    def get_class_predictions(self, X):
        z = self.forward(X)
        predictions = torch.argmax(z, dim=1)
        return predictions

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def timer_start(self):
        self.start = time.time()

    def timer_end(self, action):
        print(action + " : " + str(time.time() - self.start))

    def training_step(self, batch):
        x, y, _ = batch    
        y_pred = self.forward(x)
        supervised = torch.nn.CrossEntropyLoss(reduction=loss_reduction)(output_reduction(y_pred), y)
        return supervised

class FixMatchModel(SimpleModel):
    def __init__(self, model):
        super().__init__(model)

    def training_step(self, batch):
        x, y, unlabeled = batch    

        x = x[:int(x.shape[0] * supervised_size_ratio)]
        y = y[:int(y.shape[0] * supervised_size_ratio)]
        y_pred = self.forward(x)

        supervised = torch.nn.CrossEntropyLoss(reduction=loss_reduction)(output_reduction(y_pred), y)

        batch_size = x.shape[0]
        y_pred_class = torch.argmax(y_pred.clone().detach(), dim=1)
        accuracy = (batch_size - torch.count_nonzero(y_pred_class - y)) / float(batch_size)
        
        self.batch += 1

        unsupervised = self.fix_match_loss.loss(self, unlabeled)

        return supervised * self.labeled_loss_weight + unsupervised * self.unlabeled_loss_weight, accuracy

    def __call__(self, x):
        return self.model(x)
