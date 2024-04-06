from torch import optim
import torch
import time
from fixmatch import FixMatch
import torch.nn.functional as F

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
        supervised = F.cross_entropy(y_pred, y)
        return supervised

class FixMatchModel(SimpleModel):
    def __init__(self, model):
        super().__init__(model)

    def training_step(self, batch):
        x, y, unlabeled = batch    
        y_pred = self.forward(x)

        supervised = F.cross_entropy(y_pred, y)

        batch_size = x.shape[0]
        y_pred_class = torch.argmax(y_pred.clone().detach(), dim=1)
        accuracy = (batch_size - torch.count_nonzero(y_pred_class - y)) / float(batch_size)
        
        self.batch += 1

        unsupervised = self.fix_match_loss.loss(self, unlabeled)

        return supervised * self.labeled_loss_weight + unsupervised * self.unlabeled_loss_weight, accuracy

    def __call__(self, x):
        return self.model(x)
