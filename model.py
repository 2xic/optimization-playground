from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
import time

# just using the example model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

DEBUG = False


def debug_assert(condition):
    if DEBUG:
        return condition
    return True


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection_in = nn.Linear(100, 64)
        self.projection_out = nn.Linear(64, 100)

    def forward(self, x):
        x = F.relu(self.projection_in(x))
        x = self.projection_out(x)
        return x


class SimClrModel(pl.LightningModule):
    def __init__(self, model, projection, debug=False):
        super().__init__()
        self.model = model
        self.projection = projection
        self.debug = debug

    def forward(self, X):
        return self.projection(self.model(X))

    def training_step(self, batch, batch_idx):
        x, y = batch    

        print(x.device)

        self.timer_start()
        z_k_1 = self.forward(x)
        z_k_2 = self.forward(y)
        self.timer_end("Forward")

        N = (x.shape[0])

#        z = []
#        for i in range(N):
#            z.append(z_k_1[i])
#            z.append(z_k_2[i])

        z_fast = torch.zeros(N * 2, z_k_1.shape[-1])
        z_fast[::2, :] = z_k_1 
        z_fast[1::2, :] = z_k_2

#        assert len(z) == 2*N
        assert debug_assert(torch.allclose(z_k_1, z_k_2) ==
                            False), "Most likely something wrong"
        assert debug_assert((z_k_1).isnan().any() == False), z_k_1
        assert debug_assert((z_k_2.isnan()).any() == False), z_k_2

#        s = torch.zeros((2*N, 2*N)).type_as(x)
        total_sim = torch.zeros(1).type_as(x)
        self.timer_start()
        def fast_sim(Z):
            z_norm = Z / Z.norm(dim=1)[:, None]
            z_norm = Z / Z.norm(dim=1)[:, None]
            res = torch.mm(z_norm, z_norm.transpose(0,1))
            return res
        s = fast_sim(z_fast)
        """
        for i in range(2 * N):
            for j in range(2*N):
                results: torch.Tensor = (
                    z[i].T @ z[j]) / (torch.norm(z[i]) * torch.norm(z[j]))
                # ^ this always output a value that is the same
                # most likely something buggy with the implementation I wrote :)
                # print(pairwise_cosine_similarity(z[i].reshape(1, -1), z[j].reshape(1, -1)))
                # or maybe not, I get the same results as pairwise_cosine_similarity
              #  print(results.mean(dim=-1))

                # me testing a more "stable" loss
                #                s[i][j] = torch.relu((z[i] - z[j])).sum(dim=-1)
                s[i][j] = results.mean(dim=-1)

                total_sim += results.sum(dim=-1)
        """
        self.timer_end("SIm array")

        temperature = 0.5
        def loss(i, j):
            return -torch.log(
                torch.exp(s[i][j] / temperature)
            ) / (
                (torch.sum(
                 torch.exp(s[i] / temperature)
                 # torch.tensor([
                 #    torch.exp(s[i] / temperature) for k in range(N) if k != i
                    # ])
                 )
                 - torch.exp(s[i][i] / temperature)
                 )
            )

        def loss_compute(k): return loss(2 * k - 1, 2*k) + loss(2*k, 2 * k - 1)

        self.timer_start()
        loss_value = loss_compute(1).type_as(x)
        for k in range(2, N):
            loss_value += loss_compute(k).type_as(x)

        self.timer_end("Loss calculation")

        loss_value *= 1 / 2 * N

        self.log("train_loss", loss_value)
        self.log("total_sim", total_sim)
        self.log("total_sim_avg", total_sim / 4 * N)

        return loss_value

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def timer_start(self):
        self.start = time.time()

    def timer_end(self, action):
        print(action + " : " + str(time.time() - self.start))


class SimpleModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = nn.functional.cross_entropy(z, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = nn.functional.cross_entropy(z, y)
        batch_size = x.shape[0]
        predictions = torch.argmax(z, dim=1)
        # if all are non zero accuracy = batch size
        accuracy = batch_size - torch.count_nonzero(predictions - y)

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy / float(batch_size))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
