
from dataloader import Cifar10Dataloader
from model import Net, Projection, SimpleModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transfer_learning import TransferLearning

"""
base model that we are comparing against.
"""


base_encoder = Net()

train_loader = DataLoader(Cifar10Dataloader(), batch_size=4,
                          shuffle=True, num_workers=4)
test_set = DataLoader(Cifar10Dataloader(test=True), batch_size=4,
                      shuffle=True, num_workers=4)
model = SimpleModel(
    base_encoder,
)

trainer = pl.Trainer(limit_train_batches=500, max_epochs=30)

trainer.fit(model=model, train_dataloaders=train_loader)
trainer.test(model=model, dataloaders=test_set)
