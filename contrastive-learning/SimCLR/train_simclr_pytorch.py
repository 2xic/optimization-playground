"""
Without pytorch lightning
"""

from dataloader import SimClrCifar100Dataloader
from model import Net, Projection, SimClrModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from transfer_trained_simclr import test_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

base_encoder = Net().to(device)
projection = Projection().to(device)

model = SimClrModel(
    base_encoder,
    projection
)

train_loader = DataLoader(SimClrCifar100Dataloader(),
                          batch_size=128,
                          shuffle=True, 
                          num_workers=2)
optimizer = model.configure_optimizers()

for index, batch in enumerate(train_loader):
    optimizer.zero_grad()
    loss = model.training_step(batch)
    loss.backward()
    optimizer.step()

    if index % 10 == 0:
        print(loss)

#trainer = pl.Trainer(
#    accelerator="gpu",
#    devices=1,
#    limit_train_batches=500,
#    max_epochs=30,
#    enable_checkpointing=True,
#    default_root_dir="./checkpoints"
#)

#trainer.fit(model=model, train_dataloaders=train_loader)

test_model(model=model)
