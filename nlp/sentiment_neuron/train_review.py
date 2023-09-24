from model import Model, Config
from vocab import Vocab
import torch.optim as optim
from dataset import Dataset
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = Vocab()
max_size = 400
sentences = Dataset().get_text(max_size)
dataset = dataset.process_dataset(sentences).lock()

config = Config(
    tokens=dataset.get_vocab_size(),
    padding_index=dataset.PADDING_IDX,
    sequence_size=1
)
print(config)

model = Model(config).to(device)
print("decoded=",dataset.decode([2]))
batch_size = 32
optimizer = optim.Adam(model.parameters())
print("Starting to train ...")
"""
It is a bit slow to start with, but it learns after a few 100 epochs
"""
for epoch in range(1_000):
    loss = 0
    for index in range(0, len(sentences), batch_size):
        x, y = dataset.get_dataset(sentences[index:index + batch_size], device=device)
        x_decoded, y_decoded = dataset.decode(x.reshape(-1).tolist()), dataset.decode(y.reshape(-1).tolist())
        loss += model.fit(x, y, debug=True)

        if index % batch_size * 10 == 0:
            print(f"loss {loss} epoch: {epoch}, epoch progress {index} / {len(sentences)}")
            # batching together to get more epoch like flow
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = 0

    predict_vector = dataset.get_encoded(x_decoded[:4], device=device)
    output = model.predict(predict_vector, debug=False)
    print("predicted", dataset.decode(output))
    print("truth", x_decoded[:max_size])
    print("")
    model.save()
