import torch
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab

SEQUENCE_LENGTH = 3
source_vocab = SimpleVocab()


X, y = get_document_dataset(source_vocab, [
    "hello world, this is just some random text to verify if the model can learn something."
], SEQUENCE_LENGTH)
source_vocab.lock()

config = Config(
    vocab_size=source_vocab.size,
    embedding_dim=32,
    transformer_layers=1,
    attention_heads=4,
    dropout=0.05,
    feed_forward=128,
    padding_index=source_vocab.vocab.PADDING_IDX,
    sequence_length=SEQUENCE_LENGTH
)
model = GptTransformerModel(config)
epochs = 1024

optimi = torch.optim.Adam(
    model.parameters(),
    lr=13e-4
)

count_full_win = 0
for _ in range(1024):
    optimi.zero_grad()
    accuracy = 0
    sum_loss = 0
#    for i in range(X.shape[0]):
    output = model(X)
    loss = torch.nn.functional.cross_entropy(
        output.reshape((output.shape[0], -1)),
        y.reshape((-1)),
        ignore_index=-1
    )
    accuracy += (
        torch.argmax(output, dim=1) == y.reshape((-1))
    ).sum()
    loss.backward(
        retain_graph=True
    )
    sum_loss += loss.item()
        
    optimi.step()
    accuracy_pct = accuracy / (y.shape[0] * y.shape[1]) * 100
    print(accuracy_pct, sum_loss, count_full_win)
#    raw = model.rollout(
#        source_vocab.encode("hello"), 
#        2,
#        sampling="argmax"
#    )
    raw = torch.argmax(
        model.raw_forward(X)[:, ::3, :],
        dim=1
    ).tolist()
    print(raw)
    print(source_vocab.decode(raw))
    print("")
    if int(accuracy_pct) > 99:
        count_full_win += 1
