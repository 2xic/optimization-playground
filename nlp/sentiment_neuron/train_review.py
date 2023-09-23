from model import Model, Config
from vocab import Vocab
import torch.optim as optim
from dataset import Dataset
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = Vocab()
sentences = Dataset().get_text()
dataset = dataset.process_dataset(sentences)

config = Config(
    tokens=dataset.get_vocab_size(),
    padding_index=dataset.PADDING_IDX,
    sequence_size=1
)
print(config)

model = Model(config).to(device)
batch_size = 32
optimizer = optim.Adam(model.parameters())
print("Starting to train ...")
for epoch in range(100):
    for index in range(0, len(sentences), batch_size):
        x, y = dataset.get_dataset(sentences[index:index + batch_size], device=device)
        print(x)
        optimizer.zero_grad()
        loss = model.fit(x, y)
        loss.backward()
        optimizer.step()
        if index % batch_size * 5 == 0:
            print(f"loss {loss} epoch: {epoch}, epoch progress {index} / {len(sentences)}")

    model.save()
    for i in ["hello", "this", "gift", "I", sentences[0].split(" ")[0], sentences[1].split(" ")[0]]:
        predict_vector = dataset.get_dataset([
            f"{i} "
        ], device=device)[0]
        print(predict_vector)
        output = model.predict(predict_vector)
        print(output)
        print(dataset.decode(predict_vector.tolist()[0] + output))
    