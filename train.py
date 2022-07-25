
from dataloader import Cifar10Dataloader
from model import Net, Projection, SimpleModel
from torch.utils.data import  DataLoader
import pytorch_lightning as pl

"""
base model that we are comparing against.
"""


base_encoder = Net()
projection = Projection()

model = SimpleModel(
    base_encoder,
    projection
)

train_loader = DataLoader(Cifar10Dataloader(), batch_size=4,
                        shuffle=True, num_workers=4)
test_set = DataLoader(Cifar10Dataloader(test=True), batch_size=4,
                        shuffle=True, num_workers=4)

trainer = pl.Trainer(limit_train_batches=500, max_epochs=400)

trainer.fit(model=model, train_dataloaders=train_loader)
trainer.test(model, dataloaders=test_set)
