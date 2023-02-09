from datasource import encode, decoder
import torch
import random
from model_embedding import Embedding
from model_embedding_lstm import EmbeddingLstm

def get_batch():
    X = None
    y = None

    for i in range(32):
        query = f"10 + {random.randint(0, 30)}"
        if i == 0:
            X = torch.tensor(encode(query, size=size)).reshape((1, -1))
            y = torch.tensor(encode(str(eval(query)), size=size)).reshape((1, -1))
        else:
            X = torch.concat((
                X,
                torch.tensor(encode(query, size=size)).reshape((1, -1))
            ))
            y = torch.concat((
                y,
                torch.tensor(encode(str(eval(query)), size=size)).reshape((1, -1))
            ))
    return (X, y)

models = [
    Embedding(),
    EmbeddingLstm()
]

for model in models:
    size = model.input_size


    optimizer = torch.optim.Adam(model.parameters())
    X, y = get_batch()
    for i in range(10_000):
        y_predicted = model(X)
        absolute_error = torch.nn.L1Loss()(y_predicted, y)
        
        if i % 100 == 0:
            X, y = get_batch()
            print(absolute_error)

        optimizer.zero_grad()
        absolute_error.backward()
        optimizer.step()

    X, y = get_batch()

    output = model(X[:5])
    for (predicted, expected) in zip(decoder(output, size=size), decoder(y, size=size)):
        print((predicted, expected))
