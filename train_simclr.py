
from dataloader import SimClrCifar100Dataloader
from model import Net, Projection, SimpleModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl


base_encoder = Net()
projection = Projection()

model = SimpleModel(
    base_encoder,
    projection
)

train_loader = DataLoader(SimClrCifar100Dataloader(), batch_size=32,
                          shuffle=True, num_workers=4)

trainer = pl.Trainer(limit_train_batches=500,
                     max_epochs=400,
                     enable_checkpointing=True,
                     default_root_dir="./checkpoints"
                     )

trainer.fit(model=model, train_dataloaders=train_loader)
