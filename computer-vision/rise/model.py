from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from mnist import train_loader
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn as nn
from optimization_utils.channel.Sender import send

# just using the example model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 63)
        self.fc4 = nn.Linear(63, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc3(x))
        x = (self.fc4(x))
        x = F.log_softmax(x, dim=1)
        #x = F.softmax(x, dim=1)
        return x

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Net()
        self.batch_idx = 0

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def timer_start(self):
        self.start = time.time()

    def timer_end(self, action):
        print(action + " : " + str(time.time() - self.start))

    def training_step(self, batch, batch_indx):
        x, y = batch    
        y_pred = self.forward(x)
        loss = torch.nn.NLLLoss()(y_pred, y)
        if self.batch_idx % 100 == 0:
            y_pred = (torch.argmax(y_pred, dim=1))
            send("http://localhost:8080",
            metadata={
                "y": list([i.item() for i in y]),
                "y_pred": list([i.item() for i in y_pred])
            },
             accuracy=((y_pred == y).sum()/y_pred.shape[0]).item() )#, on_epoch=True) 

        self.batch_idx += 1
        return loss

    def test_step(self, batch, batch_indx):
        x, y = batch    
        y_pred = torch.argmax(self.forward(x), dim=1)
        print(((y_pred == y).sum()/y_pred.shape[0]).item())
        return {
            "acc": ((y_pred == y).sum()/y_pred.shape[0]).item()
        }

    def on_epoch_end(self) -> None:
        return super().on_epoch_end()

if __name__ == "__main__":
    model = SimpleModel()
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=1,
    )
    if False:
        path = "/home/brage/Desktop/paper-zoo/cv/rise/lightning_logs/version_21/checkpoints/epoch=4-step=2345.ckpt"
        model.load_from_checkpoint(path)
        model.eval()
        trainer.test(model, train_loader)
    else:
        trainer.fit(model, train_loader)
        model.eval()
#        trainer.test(model, train_loader)
        torch.save(model.model.state_dict(), "model.pkt")
