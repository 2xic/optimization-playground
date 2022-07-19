from base64 import decode
from itertools import accumulate
from dataset import *
from iterate import Iterator
from model import DecoderModel, EncoderModel

dataloader = Wmt16Dataloader(
    try_to_overfit=True
)
input_size = 100
encoder = EncoderModel(input_size, vocab_size=dataloader.vocab_size)
decoder = DecoderModel(input_size, vocab_size=dataloader.vocab_size)
iterator = Iterator(
    encoder=encoder,
    decoder=decoder,
    beginning_token=dataloader.beginning_token,
    end_token=dataloader.end_token
)

# TODO: Fix device usage, should use gpu.

print(dataloader.beginning_token)
print(dataloader.end_token)
print(dataloader.vocab_size)

# TESTING
LOG_INTERVAL = 1

for index,(X, y) in enumerate(dataloader):
    X_tensor = torch.zeros(1, input_size, dtype=torch.long)
    y_tensor = torch.zeros(1, input_size, dtype=torch.long)
    
    for index, i in enumerate(X[:input_size-1]):
        X_tensor[0][index] = i
    X_tensor[0][index + 1:] = dataloader.end_token
    for index, i in enumerate(y[:input_size-1]):
        y_tensor[0][index] = i
    y_tensor[0][index + 1:] = dataloader.end_token

    error, predicted, accumulated = iterator.iterate(X_tensor, y_tensor)

    if index % LOG_INTERVAL == 0 and index > 0:
        print(f"Index: {index}")
        print("Predicted / vs Expected")
        print(predicted)
        print(dataloader.decode(predicted))
        print(dataloader.decode(y))
        print(f"Loss {error.item()}, accumulated {accumulated}")
        print("")

