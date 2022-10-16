
from tkinter.tix import Tree
from dataloader import SimClrCifar100Dataloader
from model import Net, Projection, SimClrModel
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
base_encoder = Net()
projection = Projection()

model = SimClrModel(
    base_encoder,
    projection,
    debug=Tree
)


X, _ = SimClrCifar100Dataloader()[0]
_, Z = SimClrCifar100Dataloader()[1]
_, y = SimClrCifar100Dataloader()[2]

X, y, Z = X.reshape((1, ) + X.shape), y.reshape((1, ) + y.shape), Z.reshape((1, ) + Z.shape)

z, z2, z3 = None, None, None
for i in range(100):
    z = model.forward(X)
    z2 = model.forward(y)
    z3 = model.forward(Z)
    
   # assert torch.allclose(z, z2)
    break

plt.figure()


f, axes = plt.subplots(3,1) 
axes[0].imshow(X[0].permute(1, 2, 0))
axes[1].imshow(y[0].permute(1, 2, 0))
axes[2].imshow(Z[0].permute(1, 2, 0))

plt.show()

results: torch.Tensor = (z.T @ z2) / (torch.norm(z) * torch.norm(z2))
print(results.mean())

results: torch.Tensor = (z.T @ z3) / (torch.norm(z) * torch.norm(z3))
print(results.mean())

#plt.show()

#print(z)
#print(z2)



