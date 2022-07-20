from base64 import decode
from itertools import accumulate
from dataset import *
from get_device import get_device
from iterate import Iterator
from model import DecoderModel, EncoderModel

DEVICE = get_device(gpu=True)

train_dataloader, _, dataloader = get_data_loader(
    try_to_overfit=5
)
input_size =  15 # 100
encoder = EncoderModel(input_size, vocab_size=dataloader.vocab_size, device=DEVICE).to(DEVICE)
decoder = DecoderModel(input_size, vocab_size=dataloader.vocab_size, device=DEVICE).to(DEVICE)
iterator = Iterator(
    encoder=encoder,
    decoder=decoder,
    beginning_token=dataloader.beginning_token,
    end_token=dataloader.end_token,
    device=DEVICE,
)
parameters = lambda x: sum(p.numel() for p in x.parameters())

print(dataloader.beginning_token)
print(dataloader.end_token)
print(dataloader.vocab_size)
print(encoder)
#print(parameters(encoder))
#exit(0)

# TESTING
LOG_INTERVAL = 1
ITERATION = 0

for train_epochs in range(1_000):
    for index in range(len(dataloader)):
        (X, y) = dataloader[index]
        X_tensor = torch.zeros(1, input_size, dtype=torch.long, device=DEVICE)
        y_tensor = torch.zeros(1, input_size, dtype=torch.long, device=DEVICE)
        
        for index, i in enumerate(X[:input_size-1]):
            X_tensor[0][index] = i
        X_tensor[0][index + 1:] = dataloader.end_token
        for index, i in enumerate(y[:input_size-1]):
            y_tensor[0][index] = i
        y_tensor[0][index + 1:] = dataloader.end_token

        error, predicted, accumulated = iterator.iterate(X_tensor, y_tensor)

        if ITERATION % LOG_INTERVAL == 0 and index > 0:
            print(f"Index: {ITERATION}")
            print("Predicted / vs Expected")
            print(predicted)
            print(X)
            print(y)
            print(dataloader.decode(X))
            print("Predicted", dataloader.decode(predicted))
            print("Expected", dataloader.decode(y))
            print(f"Loss {error.item()}, accumulated {accumulated}")
            print("")
        ITERATION += 1

