from base64 import decode
from dataset import *
from iterate import Iterator
from model import DecoderModel, EncoderModel

dataloader = Wmt16Dataloader()
input_size = 100
encoder = EncoderModel(input_size, vocab_size=dataloader.vocab_size)
decoder = DecoderModel(input_size, vocab_size=dataloader.vocab_size)
iterator = Iterator(
    encoder=encoder,
    decoder=decoder,
    beginning_token=dataloader.beginning_token,
    end_token=dataloader.end_token
)

print(dataloader.beginning_token)
print(dataloader.end_token)
print(dataloader.vocab_size)
#exit(0)

for X, y in dataloader:
    X_tensor = torch.zeros(1, input_size, dtype=torch.long)
    y_tensor = torch.zeros(1, input_size, dtype=torch.long)
    
    for index, i in enumerate(X[:input_size-1]):
        X_tensor[0][index] = i
    X_tensor[0][index + 1:] = dataloader.end_token
    for index, i in enumerate(y[:input_size-1]):
        y_tensor[0][index] = i
    y_tensor[0][index + 1:] = dataloader.end_token
        
    error = iterator.iterate(X_tensor, y_tensor)
#    print(error)
