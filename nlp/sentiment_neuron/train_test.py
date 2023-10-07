"""
Mining example to make sure the model actually does something reasonable.
"""

from model import Model, Config
from vocab import Vocab
import torch.optim as optim
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = Vocab()
x, y = dataset.get_dataset([
    "hello world, I hope you are doing well today",
    "this is some text about text",
    "excellent! wonderful"
], device=device)
assert len(x.shape) == 2
config = Config(
    tokens=dataset.get_vocab_size(),
    padding_index=dataset.PADDING_IDX,
    sequence_size=1
)
print(config)
dataset.save()

model = Model.load() 
model = Model(config).to(device) if model is None else model
optimizer = optim.Adam(model.parameters())
for i in range(100):
    optimizer.zero_grad()
    loss = model.fit(x, y, debug=True)
    loss.backward()
    optimizer.step()

    print(f"loss {loss}")
    model.save()

for i in ["hello", "this", "gift", "I"]:
    predict_vector = dataset.get_encoded(f"{i}", device=device)
    output = model.predict(predict_vector, debug=False)
    print(dataset.decode(output))
    print("")
model.save()
