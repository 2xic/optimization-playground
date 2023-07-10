import torch
from transformer_encoder import TestTransformerEncoder


seq_size = 4
a = torch.tensor([
    [0, 1, 2, 3][:seq_size],
    [0, 1, 2, 3][:seq_size]
])
b = torch.tensor([
    [2,2,2,2][:seq_size],
    [2,2,2,2][:seq_size]
])

model = TestTransformerEncoder(
    ntoken=4,
    device=torch.device('cpu')
)
optimizer = torch.optim.Adam(model.parameters())

for i in range(1_000):
    optimizer.zero_grad()
    loss = model.fit(a, b)
    loss.backward()
    optimizer.step()
    print(loss)

print(model.predict(a))