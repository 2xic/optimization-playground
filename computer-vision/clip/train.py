from dataset import Flickr8Parser
from torch.utils.data import DataLoader
from vocab import Vocab
from model import Model
import torch
import torch.optim as optim
from plot import plot

dataset = Flickr8Parser(
    max_dataset_size=100
)
vocab = Vocab()
dataset_loader = DataLoader(dataset,batch_size=32, shuffle=True)

model = Model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

normalize = lambda x:  torch.nn.functional.normalize(x, dim=1)

for i in range(1_00):
    sum_loss = 0
    for index, (x, y) in enumerate(dataset_loader):
        encoded = vocab.encode_sentence(y)
        image, text, scale = model(
            x,
            encoded
        )

        logits_per_image = scale * image @ text.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(x.shape[0])
        loss_i = torch.nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_t = torch.nn.CrossEntropyLoss()(logits_per_text, labels)
        loss = (loss_i + loss_t)/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    print(i, sum_loss)

X, y = next(iter(dataset_loader))
plot(
    model=model,
    x=X,
    y=y,
    vocab=vocab
)
