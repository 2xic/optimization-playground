"""
Saw this library and it look awesome

https://github.com/huggingface/accelerate/tree/main
https://huggingface.co/docs/accelerate/index

"""
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.dataloaders.Mnist import get_dataloader

train, test = get_dataloader()
accelerator = Accelerator()
device = accelerator.device

model = BasicConvModel().to(device)
optimizer = torch.optim.Adam(model.parameters())

# automatically selects cuda, nice.
print(device)
model, optimizer, data = accelerator.prepare(model, optimizer, train)

model.train()
for epoch in range(10):
    for source, targets in data:
        source = source.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(source)
        loss = F.cross_entropy(output, targets)

        accelerator.backward(loss)

        optimizer.step()
