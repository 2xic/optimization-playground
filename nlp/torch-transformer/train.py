from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.Transformer import TransformerModel, Config
import torch
import torch.optim as optim

SEQUENCE_LENGTH = 10
source_vocab = SimpleVocab()
target_vocab = SimpleVocab()
X = torch.concat([
    source_vocab.get_tensor("<START>", sequence_length=SEQUENCE_LENGTH),
    source_vocab.get_tensor("<START> example", sequence_length=SEQUENCE_LENGTH),
    source_vocab.get_tensor("<START> example text", sequence_length=SEQUENCE_LENGTH),
    source_vocab.get_tensor("<START> example text that", sequence_length=SEQUENCE_LENGTH),
    source_vocab.get_tensor("<START> example text that should", sequence_length=SEQUENCE_LENGTH),
    source_vocab.get_tensor("<START> example text that should return", sequence_length=SEQUENCE_LENGTH),
    source_vocab.get_tensor("<START> example text that should return <END>", sequence_length=SEQUENCE_LENGTH),
], dim=0)
y = torch.concat([
    target_vocab.get_tensor("", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that should", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that should return", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that should return <END>", sequence_length=SEQUENCE_LENGTH),
], dim=0)
y_target = torch.concat([
    target_vocab.get_tensor("example", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that should", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that should return", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that should return <END>", sequence_length=SEQUENCE_LENGTH),
    target_vocab.get_tensor("example text that should return <END>", sequence_length=SEQUENCE_LENGTH),
], dim=0)
assert X.shape[0] == y.shape[0]

source_vocab.lock()    
target_vocab.lock()


config = Config(
    encoder_vocab=source_vocab.size,
    decoder_vocab=target_vocab.size,
    embedding_dim=8,
    transformer_layers=1,
    attention_heads=8,
    dropout=0.1,
    feed_forward=128,
    padding_index=source_vocab.vocab.PADDING_IDX,
)
model = TransformerModel(config)
optimizer = optim.Adam(model.parameters())
for _ in range(1_000):
    optimizer.zero_grad()
    y_prediction = model(X, y)
    loss = torch.nn.CrossEntropyLoss(ignore_index=source_vocab.vocab.PADDING_IDX)(y_prediction, y_target)
    loss.backward()
    optimizer.step()
    print(loss.item())

y = model.rollout(
    X=torch.concat([
        source_vocab.get_tensor("<START>", sequence_length=SEQUENCE_LENGTH)
    ], dim=0),
    steps=8,
)
print(y)
for i in y:
    print(target_vocab.vocab.index_vocab[i.item()])

