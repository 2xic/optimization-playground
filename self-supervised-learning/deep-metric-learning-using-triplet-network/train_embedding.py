from collections import defaultdict
from model import SimpleModel
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import random
from triplet_model import TripletModel
import torch

#model = SimpleModel()
model = TripletModel()
optimizer = Adam(model.parameters())

train_ds = MNIST("./", train=True, download=True,
                 transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)


for epoch in range(3):
    for batch_size, (X, y) in enumerate(train_loader):
        # should have done more effetely batch generation,but it's late
        batch_labels = defaultdict(int)
        index_labels = defaultdict(list)
        for index, i in enumerate(y):
            batch_labels[i.item()] += 1
            index_labels[i.item()].append(index)

        batch_labels = {
            key: batch_labels[key] for key in batch_labels if batch_labels[key] > 1
        }
        if len(batch_labels) == 0:
            continue
        

        loss = 0
        while len(batch_labels):
            idx = random.randint(0, len(batch_labels) -1)
            label = list(batch_labels.keys())[idx]
            indexes = index_labels[label]

      #      for _ in range(8):
            idx = random.randint(0, 31)
            other_sample = y[idx]
            if other_sample.item() != label:
                negative_sample = X[idx]
                plus = random.sample(indexes, 2)
                x_ref = X[plus[0]]
                x_plus = X[plus[1]]

                output = model(
                    negative_sample,
                    x_ref,
                    x_plus
                )
                loss += output
            del batch_labels[label]
        
        if batch_size % 100 == 0:
            print(loss)

        if torch.is_tensor(loss):
            model.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"Epoch {epoch}")

torch.save({
    'model_state_dict': model.state_dict(),
}, "model_state")

