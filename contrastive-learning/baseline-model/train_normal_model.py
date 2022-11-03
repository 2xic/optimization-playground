
from dataloader import Cifar10Dataloader
from model import Net, SimpleModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl


"""
base model that we are comparing against.
"""
base_encoder = Net()
batch_size = 64

train_loader = DataLoader(Cifar10Dataloader(), batch_size=batch_size,
                          shuffle=True, num_workers=2)
test_set = DataLoader(Cifar10Dataloader(test=True), batch_size=batch_size, num_workers=2)
model = SimpleModel(
    base_encoder,
)

trainer = pl.Trainer(limit_train_batches=500, accelerator="gpu", max_epochs=30)

trainer.fit(model=model, train_dataloaders=train_loader)
trainer.test(model=model, dataloaders=test_set)
