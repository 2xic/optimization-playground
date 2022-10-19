from json import tool
import torch
from torch.utils.data import Dataset
from dataset import get_quadratic_function_error
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class SimpleDataset(Dataset):
	def __init__(self) -> None:
		self.xor = [
			[[0, 0], [1, 0 ]], 
			[[1, 1], [1, 0]], 

			[[0, 1], [0, 1]], 
			[[1, 0], [0, 1]], 
		]

		# 0 is 0
		# 1 is 1
		self.xy = [
			[[0, 0], [1, 0 ]], 
			[[0, 1], [0, 1]], 
		]


		single = False
		if single:
			for index in range(len(self.xy)):
				self.xy[index][1] = self.xy[index][1].index(max(self.xy[index][1]))

	def __len__(self):
		return len(self.xy)

	def __getitem__(self, idx):
		row = self.xy[idx]

		x = torch.tensor(row[0]).float()
		y = torch.tensor(row[1]).float()

		return x, y

class SimpleModel(pl.LightningModule):
	def __init__(self, optim, lr):
		super(SimpleModel, self).__init__()

		self.lr = lr
		self.optim = optim
		self.seq = nn.Sequential(*[
			nn.Linear(2, 4),
			torch.nn.Sigmoid(),
			nn.Linear(4, 4),
			nn.Linear(4, 2),
			torch.nn.Softmax(dim=1),
		])

	def forward(self, x):
		return self.seq(x)

	def configure_optimizers(self):
		return self.optim(self.parameters(), lr=self.lr)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.forward(x)
		#print(y_hat, y)
		#loss = torch.nn.CrossEntropyLoss()(y_hat, y) 
		#loss = (y_hat -y).sum() #torch.nn.CrossEntropyLoss()(y_hat, y.long()) 
		loss = ((y_hat -y)**2).sum()


		self.log("val_loss", loss)
		return loss

def train(optim=torch.optim.Adam, lr=1e-3, max_epochs=3_00):
	model = SimpleModel(optim, lr)
	dataset = SimpleDataset()
	train_loader = DataLoader(dataset,
                        batch_size=8,
                        shuffle=True, 
                        num_workers=8)
	trainer = pl.Trainer(
		callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=25)],
		max_epochs=max_epochs,
	)
	trainer.fit(model=model, train_dataloaders=train_loader)

	for i in range(len(dataset)):
		X, y = dataset[i]
		print(
			model(X.reshape((1, -1))),
			y
		)


if __name__ == "__main__":
	train()
	