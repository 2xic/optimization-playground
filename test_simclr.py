from model import Net, Projection, SimpleModel, SimClrModel
from transfer_learning import TransferLearning
import pytorch_lightning as pl

from dataloader import Cifar100Dataloader, Cifar10Dataloader, SimClrCifar100Dataloader
from model import Net, Projection, SimpleModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
from dataloader import transform


def test_model(model):
    transfer_model = TransferLearning(model, output_size=10)

    dataset = Cifar10Dataloader()
    train_loader = DataLoader(dataset, batch_size=32,shuffle=True, num_workers=4)

    test_dataset = Cifar10Dataloader(test=True)
    test_loader = DataLoader(test_dataset, batch_size=32,shuffle=True, num_workers=4)

    trainer = pl.Trainer(limit_train_batches=500,max_epochs=30)
    trainer.fit(model=transfer_model, train_dataloaders=train_loader)
    trainer.test(model=transfer_model, dataloaders=test_loader)

if __name__ == "__main__":
    model = Net()
    projection = Projection()
    loaded_checkpoint = SimClrModel.load_from_checkpoint(
        checkpoint_path="checkpoints/lightning_logs/version_5/checkpoints/epoch=99-step=50000.ckpt", model=model, projection=projection)
    test_model(model)
