import torch
import torch.nn as nn
from tqdm import tqdm


class MultipleGPUsTrainingLoopDistributedAccumulate:
    def __init__(self,
                 model,
                 optimizer,
                 gpu_id,
                 loss=nn.NLLLoss(),
                 update_step=32
        ):
        self.gpu_id = gpu_id
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = 0
        self.update_step = update_step
        self.is_main_gpu = self.gpu_id == 0

    def eval(self, dataloader):
        with torch.no_grad():
            (_, acc) = self._iterate(dataloader, train=False)
            return acc

    def train(self, dataloader):
        return self._iterate(dataloader, train=True)

    def _iterate(self, dataloader, train=True):
        device = self.gpu_id
        if hasattr(dataloader, 'sampler'):
            dataloader.sampler.set_epoch(self.epoch)

        dataloader = tqdm(dataloader) if self.is_main_gpu else dataloader
        total_loss, accuracy = self._iterate_dataloader(
            dataloader,
            device,
            train,
        )

        self.epoch += 1
        self._step()

        if self.gpu_id == 0:
            if self.epoch % 100 == 0:
                print(
                    f"[GPU{self.gpu_id}] Epoch {self.epoch} | Loss {total_loss.item()}"
                )
            return (
                total_loss,
                accuracy
            )
        return None
    
    def _iterate_dataloader(self, dataloader, device, train):
        length = 0
        total_loss = torch.tensor(0.0, device=device)
        accuracy = torch.tensor(0.0, device=device)

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            y_pred = self.model(X)

            if train:
                loss = self.loss(y_pred, y)
                loss.backward()

                if 0 < batch and batch % self.update_step == 0:
                    self._step()
                if isinstance(dataloader, tqdm):
                    dataloader.set_description(f"batch: {batch}, loss: {loss}")
                total_loss += loss.item()
                self._batch_done()
            accuracy += (torch.argmax(y_pred.detach(), 1) == y).sum()
            length += X.shape[0]
        accuracy = (accuracy / length) * 100
        return total_loss, accuracy
    
    def _step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _batch_done(self):
        pass
