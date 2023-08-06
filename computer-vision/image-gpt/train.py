"""
Train to predict next pixel from the previous batch of pixels
"""
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
# from optimization_playground_shared.nlp.Transformer import TransformerModel, Config
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from torchvision.utils import save_image

train, _ = get_dataloader(
    batch_size=4
)
vocab = SimpleVocab()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SEQUENCE_SIZE = 512
X_input = []
y_context = []
y_target = []


for X, _ in train:
    X = X.reshape(X.shape[0], -1)
    for i in X:
        vector = ["<START>"]
        for j in i:
            vector.append(str(round(j.item(), 1)))
        output = []
        for index in range(0, len(vector) - 1):
            x = vocab.get_tensor(None, sequence_length=SEQUENCE_SIZE) if len(output) == 0 else vocab.get_tensor(
                " ".join(vector[max(0, index - SEQUENCE_SIZE):index]), sequence_length=SEQUENCE_SIZE)
       #     context = vocab.get_tensor(None, sequence_length=SEQUENCE_SIZE) if len(output) == 0 else output[-1]
            y = vocab.get_tensor(" ".join(
                vector[max(0, index - SEQUENCE_SIZE):index + 1]), sequence_length=SEQUENCE_SIZE)

            if len(output):
                X_input.append(x)
                y_target.append(y)
            output.append(y)
  #      break
    break

print(vocab.size)
for idx in range(128):
    print(X_input[idx])
#    print(y_context[idx])
    print(y_target[idx])
    print("")
#exit(0)


X_input = torch.concat(X_input, dim=0).to(device)
y_target = torch.concat(y_target, dim=0).to(device)


config = Config(
    vocab_size=vocab.size,
    embedding_dim=8,
    transformer_layers=1,
    attention_heads=8,
    dropout=0,
    feed_forward=256,
    padding_index=vocab.vocab.PADDING_IDX,
)
model = GptTransformerModel(config).to(device)

dataloader = get_raw_dataloader((
    X_input,
    y_target
), batch_size=256)
optimizer = optim.Adam(model.parameters())
for epoch in range(50):
    for mini_x,  mini_y in dataloader:
        optimizer.zero_grad()
        y_prediction = model(mini_x, mini_x)
     #   print(y_prediction.argmax(dim=1))
     #   print(mini_y)
        loss = torch.nn.CrossEntropyLoss(
            ignore_index=vocab.vocab.PADDING_IDX)(y_prediction, mini_y)
        loss.backward()
        optimizer.step()
      #  print(epoch, loss.item())

    with torch.no_grad():
        """
        Output image
        """
        y = model.rollout(
            seed=vocab.get_tensor(" ".join(vector[:SEQUENCE_SIZE]), sequence_length=SEQUENCE_SIZE).reshape(-1)[:512],
            steps=28 * 28,
            device=device
        )
        output = []
        for i in y:
            print(i.item(), end=" ")
            try:
                output.append(float(vocab.vocab.index_vocab[i.item()]))
            except Exception as e:
                output.append(0)
                print("Unknown pixel value")
        tensor = torch.tensor(output).reshape((28, 28))
#        print(tensor)
#        save_image(X[0].reshape((28, 28)), 'mnist.png')
        save_image(tensor, f'debug/output_{epoch}.png')
