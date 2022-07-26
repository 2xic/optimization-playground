
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
    limit_train_batches=100,
    max_epochs=5, #30,
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

- forgot to add the loss value normalization
    - retraining with 10 epochs for simclr + 30 for feature model
    - It still stops at 0.344 hm.
    - accuracy around 0.17%
- fixed an error with the sign of the loss and temperature location
    - retraining with 10 epochs for simclr + 30 for feature model
    - now the loss is stuck even more. 
    - 10 % acc
- found a bug with the wait the Z was constructed. Looked at the batch input size (X,y )    
    instead of the shape batch tensor size. Ops
    - retraing now.
    - loss is decreasing, and I trained for a few epochs.
    - accuaracy 0.18%
- will try again tomorrow to train for a longer time period, and see what happens :)
    
"""
test_model(model=model)
