from Model import FixMatchModel, Net
from Dataloader import Cifar10Dataloader
from torch.utils.data import DataLoader
import pytorch_lightning as pl

train_loader = DataLoader(Cifar10Dataloader(),
                          batch_size=64,
                          shuffle=True,
                          num_workers=8)
model = FixMatchModel(Net())

trainer = pl.Trainer(
    accelerator="cpu",
    limit_train_batches=500,
    max_epochs=30,  # 30,
    enable_checkpointing=True,
    default_root_dir="./checkpoints"
)

trainer.fit(model=model, train_dataloaders=train_loader)
