from statistics import mode
from simple_env_trajectory import play
from torch_model import TorchModel

model = TorchModel()

loss = play(model)
print(loss)

