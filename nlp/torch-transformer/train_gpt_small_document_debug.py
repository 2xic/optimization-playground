import torch
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset
from optimization_playground_shared.nlp.wordpiece.bpe import BPE
from optimization_playground_shared.nlp.wordpiece.bpeDocumentDecoder import get_document_dataset as get_document_dataset_bpe
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab

SEQUENCE_LENGTH = 10

vocab_type = "simple"
source_vocab = None
docs = [
    "hello world, this is just some random text to verify if the model can learn something."
]
PADDING_IDX = None
if vocab_type == "simple":
    source_vocab = SimpleVocab()
    X, y = get_document_dataset(source_vocab, docs, SEQUENCE_LENGTH)
    source_vocab.lock()
    PADDING_IDX = source_vocab.vocab.PADDING_IDX
else:
    source_vocab = BPE()
    source_vocab.add_tokens(docs).run_merge_step()
    X, y = get_document_dataset_bpe(source_vocab, docs, SEQUENCE_LENGTH)
    PADDING_IDX = source_vocab.index.padding_idx

    assert "hello" == source_vocab.decode(source_vocab.encode("hello"))
    assert "hello world" == source_vocab.decode(source_vocab.encode("hello world"))
# exit(0)

print(y)

config = Config(
    vocab_size=source_vocab.size,
    embedding_dim=32,
    transformer_layers=2,
    attention_heads=4,
    dropout=0.05,
    feed_forward=128,
    padding_index=PADDING_IDX,
    sequence_length=SEQUENCE_LENGTH
)
model = GptTransformerModel(config)
epochs = 1024

optimi = torch.optim.Adam(
    model.parameters(),
    lr=13e-4
)

count_full_win = 0
for _ in range(1024 * 8):
    optimi.zero_grad()
    accuracy = 0
    sum_loss = 0
#    for i in range(X.shape[0]):
    output = model(X).reshape((-1, config.vocab_size))
    loss = torch.nn.functional.cross_entropy(
        output,
        y.reshape((-1)),
        ignore_index=PADDING_IDX
    )
    #print(output)
    accuracy += (
        torch.argmax(output, dim=1) == y.reshape((-1))
    ).sum()
    """
    print("Predicted vs actual")
    print((
         torch.argmax(output, dim=1)
    ))
    print(
        y.reshape((-1))
    )
    """

    loss.backward()
    sum_loss += loss.item()
        
    optimi.step()
    accuracy_pct = accuracy / (y.shape[0] * y.shape[1]) * 100
    print(accuracy_pct, sum_loss, count_full_win)
    raw = torch.argmax(
        model(X[:1]).reshape((-1, config.vocab_size)),
        dim=1
    ).tolist()[::config.sequence_length]

    tokens, x_tokens = model.rollout(
        X[0].tolist(), 
        config.sequence_length,
        sampling="argmax"
    )
    print("Predicted vs actual tokens for X[0]")
    print("\t" + str(tokens[config.sequence_length:]))
    print("\t" + str(y[0]))
    print("Decoded (predicted vs actual)")
    print("\t" + source_vocab.decode(tokens))
    print("\t" + source_vocab.decode(X[0].tolist() + y[:1].tolist()[0]))
    print("")
    """
    tokens, x_tokens = model.rollout(source_vocab.encode("hello"), 32)
    print(X[:1])
    print(x_tokens[:1])
    print("Rollout:\n\t", source_vocab.decode(tokens))
    print("Forward:\n\t", source_vocab.decode(raw))
    print("")
    """
    if int(accuracy_pct) > 99:
        count_full_win += 1
