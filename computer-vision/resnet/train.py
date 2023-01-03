
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import TrainableModel
from dataloader import Cifar10Dataloader


for skip_connections in [True, False]:
    model = TrainableModel(
        skip_connections=skip_connections
    )

    train_loader = DataLoader(Cifar10Dataloader(),
                            batch_size=64,
                            shuffle=True, 
                            num_workers=8)

    trainer = pl.Trainer(
    # accelerator="gpu",
        limit_train_batches=500,
        max_epochs=5,  # 30,
        enable_checkpointing=True,
        default_root_dir="./checkpoints"
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
