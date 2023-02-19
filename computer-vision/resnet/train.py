
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import TrainableModel
from dataloader import Cifar10Dataloader


for skip_connections in [True, False]:
    train_loader = DataLoader(Cifar10Dataloader(),
                            batch_size=64,
                            shuffle=True, 
                            num_workers=8)
    test_loader = DataLoader(Cifar10Dataloader(
        test=True
    ),
                            batch_size=64,
                            shuffle=True, 
                            num_workers=8)
    model = TrainableModel(
        skip_connections=skip_connections,
        test_loader=test_loader
    )
    trainer = pl.Trainer(
         accelerator="gpu",
        limit_train_batches=500,
        max_epochs=100,
        enable_checkpointing=True,
        default_root_dir="./checkpoints"
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
