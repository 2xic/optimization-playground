
from decimal import MAX_PREC
from model import Learning2Learn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataloader import QuadraticFunctionsDataset

max_epochs = 1_000

dataset = QuadraticFunctionsDataset()
train_loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True, 
                        num_workers=8)
model = Learning2Learn(
    n_features=1,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

def test_epoch(label):
    for i in range(10):
        x, y = dataset[i]
        error = model(x, y, logging=True)
        print(label, error)

test_epoch(label="Before traning ")

trainer = pl.Trainer(
    limit_train_batches=500,
    max_epochs=max_epochs,
    enable_checkpointing=True,
    default_root_dir="./checkpoints_l2l"
)
trainer.fit(model=model, train_dataloaders=train_loader)

test_epoch(label="After traning ")
