from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import torch
import torch.optim as optim

SEQUENCE_LENGTH = 10
source_vocab = SimpleVocab()
device = torch.device('cpu')

def sliding_vocab(text):
    X = []
    y = []

    words = text.lower().split(" ")
    for i in range(1, len(words) - 1):
        X.append(source_vocab.get_tensor(" ".join(words[max(i-SEQUENCE_LENGTH, 0):i]), sequence_length=SEQUENCE_LENGTH))
        y.append(source_vocab.get_tensor(" ".join(words[i:i+1]), sequence_length=1)[0])
    return torch.concat(X), torch.concat(y)

X, y = sliding_vocab("""
The quick brown fox jumps over the lazy dog :)
""")
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
model = GptTransformerModel(config)


optimizer = optim.Adam(model.parameters())
for _ in range(1_000):
    optimizer.zero_grad()
    y_prediction = model(X, X)
    loss = torch.nn.CrossEntropyLoss(ignore_index=source_vocab.vocab.PADDING_IDX)(y_prediction, y)
    loss.backward()
    optimizer.step()

y_prediction = model(X, X)

y = model.rollout(
    seed=source_vocab.get_tensor("the", sequence_length=SEQUENCE_LENGTH).reshape(-1)[:1],
    steps=15,
    device=device,
)
for i in y:
    print(source_vocab.vocab.index_vocab[i])

