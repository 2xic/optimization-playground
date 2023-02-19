import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from optimization_utils.logging.EpochRuns import EpochRuns

def conv_layer(in_channel, out_channel, kernel_size=3, stride=1):
    return nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        bias=False,
    )

class ResidualBlock(nn.Module):
    def __init__(self, input, output, is_active=True) -> None:
        super().__init__()
        
        self.conv_1 = conv_layer(input, output)
        self.conv_2 = conv_layer(output, output)

        self.is_active = is_active

    def forward(self, x):
        residual = x

        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)

        if self.is_active:
            x += residual

        return F.relu(x)


class Net(nn.Module):
    def __init__(self, skip_connections):
        super().__init__()
        self.conv1 = conv_layer(3, 16)
        self.conv2 = ResidualBlock(16, 16, is_active=skip_connections)
        self.conv3 = ResidualBlock(16, 16, is_active=skip_connections)
        self.conv4 = ResidualBlock(16, 16, is_active=skip_connections)
        self.output = nn.Linear(16384, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        x = nn.Softmax(dim=1)(x)
        return x

class TrainableModel(pl.LightningModule):
    def __init__(self, skip_connections, test_loader):
        super().__init__()
        self.epoch_information = EpochRuns(
            "resnet" if skip_connections else "non_resnet"
        )
        self.test_loader = test_loader
        self.model = Net(skip_connections)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch

        z_k_1 = self.forward(x)
        
        loss_value = torch.nn.CrossEntropyLoss()(z_k_1, y)
    
        self.log("train_loss", loss_value)

        self.epoch_information.log(
            "train_loss", loss_value.item(), self.current_epoch,
        )
        self.epoch_information.log(
            "training_accuracy", (torch.sum(torch.argmax(z_k_1, 1) == y) / x.shape[0]).item() * 100, self.current_epoch,
        )
        return loss_value

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_end(self):
        # run an quick forward of acc on test dataset
        testing_acc = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                #device = self.model.conv1.device                
                x = x.to(torch.device('cuda'))
                y = y.to(torch.device('cuda'))
                z_k_1 = self.forward(x)
                testing_acc += (torch.sum(torch.argmax(z_k_1, 1) == y) / x.shape[0]).item() * 100
        self.epoch_information.log(
            "testing_accuracy", testing_acc, self.current_epoch,
        )
        self.epoch_information.store()
