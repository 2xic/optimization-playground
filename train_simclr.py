
from dataloader import SimClrCifar100Dataloader
from model import Net, Projection, SimClrModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from test_simclr import test_model


base_encoder = Net()
projection = Projection()

model = SimClrModel(
    base_encoder,
    projection
)

train_loader = DataLoader(SimClrCifar100Dataloader(), batch_size=32,
                          shuffle=True, num_workers=4)

trainer = pl.Trainer(
    limit_train_batches=500,
    max_epochs=10, #30,
    enable_checkpointing=True,
    default_root_dir="./checkpoints"
)

trainer.fit(model=model, train_dataloaders=train_loader)
"""
basic model gives output around 50 % at 30 epochs.
You should be able to beat that.
- tested 1 epoch simclr + 30 for feature = 0.17 acc %
- tested 50 epoch simclr + 30 for feature = 0.16 acc % 
    HM ? 
- tested 10 epochs simclr + 30 for feature (but added relu between simclr and feature )= 0.18% acc

Okay, I guess something is wrong with the simclr logic.
"""
test_model(model=model)
