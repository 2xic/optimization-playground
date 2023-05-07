import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class TrainingLoopDistributedAccumulate:
    def __init__(self, 
                 model, 
                 optimizer,
                 gpu_id
        ):
        self.gpu_id = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.optimizer = optimizer
        self.loss = nn.NLLLoss()
        self.epoch = 0
        
    def eval(self, dataloader):
        with torch.no_grad():
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

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            y_pred = self.model(X)

            if train:
                loss = self.loss(y_pred, y)
                loss.backward()
                
                if 0 < batch and batch % 16 == 0:
                    self._step()
                    
                total_loss += loss
            accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += X.shape[0]
        accuracy = (accuracy / length) * 100 
        self.epoch += 1
        self._step()

        if self.epoch % 100 == 0:
            print(f"[GPU{self.gpu_id}] Epoch {self.epoch} | Loss {total_loss.item()}")

        return (
            total_loss,
            accuracy
        )

    def _step(self):
        self.optimizer.step()      
        self.optimizer.zero_grad()
