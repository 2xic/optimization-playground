from statistics import mode
from simple_env_trajectory import play
from torch_model import TorchModel
from torch.optim import Adam
from epsilon import Epsilon

model = TorchModel()
optimizer = Adam(
    model.model.parameters(),
    lr=3e-4
)
epsilon = Epsilon()

for epoch in range(1_000):
    (loss, _, total_reward) = play(model, epsilon)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(loss, total_reward, epsilon.epsilon)
