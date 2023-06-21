from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from MainNet import MainNet, WeightLayer
from HyperNetwork import HyperNetwork
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
import torch.optim as optim

weight = WeightLayer(
    width=1,
    height=1,
    z=64
)
hyper_net = HyperNetwork(
    z=64,
)

model = MainNet(
    (1, 28, 28),
    hyper_net=hyper_net
)

optimizer = optim.Adam(model.parameters())
training_loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
)

(train, test) = get_dataloader()
# needs more time to wake up ?
for _ in range(10):
    error = training_loop.train(
        train
    )
    print(error)

print("=" * 32)
print("standard model")
model = BasicConvModel(
    (1, 28, 28)
)
optimizer = optim.Adam(model.parameters())
training_loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
)
error = training_loop.train(
    train
)
print(error)
