from model import Model, Config
from vocab import Vocab
import torch.optim as optim

dataset = Vocab()
x, y = dataset.get_dataset([
    "hello world, I hope you are doing well today",
    "this is some text about text",
    "excellent! wonderful"
])
assert len(x.shape) == 2
config = Config(
    tokens=dataset.get_vocab_size(),
    padding_index=dataset.PADDING_IDX,
    sequence_size=1
)
print(config)

model = Model(config)
if not model.load() or True:
    optimizer = optim.Adam(model.parameters())
    for i in range(100):
        optimizer.zero_grad()
        loss = model.fit(x, y)
        loss.backward()
        optimizer.step()

        print(f"loss {loss}")

for i in ["hello", "this", "gift", "I"]:
    predict_vector = dataset.get_dataset([
        f"{i} "
    ])[0]
    print(predict_vector)
    output = model.predict(predict_vector, debug=False)
    print(dataset.decode(predict_vector.tolist()[0] + output))
model.save()
