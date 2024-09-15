import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

class MultipleGPUsTrainingLoop:
    def __init__(self, 
                 model, 
                 optimizer,
                 gpu_id,
                 loss=nn.NLLLoss()
        ):
        self.gpu_id = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = 0
        self.is_main_gpu = self.gpu_id == 0

    def eval(self, dataloader):
        (_, acc) = self._iterate(dataloader, train=False)
        return acc

    def train(self, dataloader):
        return self._iterate(dataloader, train=True)

    def _iterate(self, dataloader, train=True):
        device = self.gpu_id 
        total_loss = torch.tensor(0.0, device=device)
        accuracy = torch.tensor(0.0, device=device)
        length = 0
        dataloader.sampler.set_epoch(self.epoch)

        dataloader = tqdm(dataloader) if self.is_main_gpu else dataloader

        for _, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            y_pred = self.model(X)

            if train:
                loss = self.loss(y_pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()      

                total_loss += loss

            accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += X.shape[0]
        accuracy = (accuracy / length) * 100 

        epoch = self.epoch
        self.epoch += 1

        if self.gpu_id == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Loss {total_loss.item()} | Accuracy {accuracy}")
            return (
                total_loss,
                accuracy
            )
        return None
