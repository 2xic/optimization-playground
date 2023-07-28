from transfer_learning import TransferLearning
import pytorch_lightning as pl
from dataloader import Cifar10Dataloader
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import SimClrTorch


def test_model(model: SimClrTorch):
    transfer_model = TransferLearning(model.model, output_size=10)

    dataset = Cifar10Dataloader()
    train_loader = DataLoader(dataset, batch_size=32,
                              shuffle=True, num_workers=4)

    test_dataset = Cifar10Dataloader(test=True)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=True, num_workers=4)

    trainer = pl.Trainer(limit_train_batches=500, max_epochs=30)
    trainer.fit(model=transfer_model, train_dataloaders=train_loader)
    output = trainer.test(model=transfer_model, dataloaders=test_loader)
    return output[0]["test_accuracy"]
