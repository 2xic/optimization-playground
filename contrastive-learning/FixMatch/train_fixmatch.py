from Model import FixMatchModel, Net
from Dataloader import Cifar10Dataloader
from torch.utils.data import DataLoader
import pytorch_lightning as pl

train_loader = DataLoader(Cifar10Dataloader(),
                          batch_size=256,
                          shuffle=True,
                          num_workers=2)

model = FixMatchModel(Net())
dataloader = Cifar10Dataloader()


trainer = pl.Trainer(
    accelerator="gpu",
    limit_train_batches=1000,
    max_epochs=6_00,
    enable_checkpointing=True,
    default_root_dir="./checkpoints"
)

trainer.fit(model=model, train_dataloaders=train_loader)

test_loader = DataLoader(Cifar10Dataloader(test=True),
                         batch_size=64,
                         num_workers=2
                         )
model.eval()
trainer.test(model=model, dataloaders=test_loader)
