from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader

SEQUENCE_LENGTH = 10
source_vocab = SimpleVocab()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def sliding_vocab(text):
    X = []
    y = []

    words = text.lower().split(" ")
    for i in range(1, len(words) - 1):
        X.append(source_vocab.get_tensor(" ".join(words[max(i-SEQUENCE_LENGTH, 0):i]), sequence_length=SEQUENCE_LENGTH))
        y.append(source_vocab.get_tensor(" ".join(words[i:i+1]), sequence_length=1)[0])

        if 4 * 2048 < len(X):
            print("Skipping now")
            break
    return torch.concat(X), torch.concat(y)

text = None
with open("shakespeare.txt", "r") as file:
    text = file.read()

X, y = sliding_vocab(text)
source_vocab.lock()

config = Config(
    vocab_size=source_vocab.size,
    embedding_dim=8,
    transformer_layers=1,
    attention_heads=8,
    dropout=0.1,
    feed_forward=128,
    padding_index=source_vocab.vocab.PADDING_IDX,
    sequence_size=10
)
model = GptTransformerModel(config).to(device)


dataloader = get_raw_dataloader((X, y), batch_size=2048)
optimizer = optim.Adam(model.parameters())
for epoch in range(1_000):
    sum_loss = 0
    for X, y in dataloader:
        optimizer.zero_grad()
        y_prediction = model(X.to(device))
        loss = torch.nn.CrossEntropyLoss(ignore_index=source_vocab.vocab.PADDING_IDX)(y_prediction.to(device), y.to(device))
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    print(f"{epoch} {sum_loss}")

y_prediction = model(X.to(device))

y = model.rollout(
    seed=source_vocab.get_tensor("this", sequence_length=SEQUENCE_LENGTH).reshape(-1)[:1],
    steps=15,
    device=device,
)
for i in y:
    print(source_vocab.vocab.index_vocab[i])

