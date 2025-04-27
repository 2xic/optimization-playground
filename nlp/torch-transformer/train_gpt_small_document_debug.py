import torch
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset
from optimization_playground_shared.nlp.wordpiece.bpe import BPE
from optimization_playground_shared.nlp.wordpiece.bpeDocumentDecoder import get_document_dataset as get_document_dataset_bpe
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab

SEQUENCE_LENGTH = 8

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu' \
'')
#vocab_type = "bpe"
vocab_type = "simple"
source_vocab = None
docs = [
    "hello world, this is just some random text to verify if the model can learn something."
]
#    "hello world, this is just some random text to verify if the model can learn something."
#with open("example.text", "r") as file:
#    docs.append(file.read())

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
    assert PADDING_IDX == 0


config = Config(
    vocab_size=source_vocab.size,
    embedding_dim=16,
    transformer_layers=2,
    attention_heads=4,
    dropout=0,
    feed_forward=128,
    padding_index=PADDING_IDX,
    sequence_length=SEQUENCE_LENGTH
)
model = GptTransformerModel(config).to(DEVICE)
X = X.to(DEVICE)
y = y.to(DEVICE)
epochs = 1024

optimi = torch.optim.Adam(
    model.parameters(),
    lr=13e-4
)

count_full_win = 0
for _ in range(1024 * 8):
    model.train()
    accuracy = 0
    sum_loss = 0

    output = model(X).reshape((-1, config.vocab_size))
    loss = torch.nn.functional.cross_entropy(
        output,
        y.reshape((-1)),
        ignore_index=PADDING_IDX
    )
    loss.backward()
    sum_loss += loss.item()
    optimi.step()
    optimi.zero_grad()
    # Eval now to make things are stable
    model.eval()

    output = model(X).reshape((-1, config.vocab_size))
    predicted_argmax = torch.argmax(output, dim=1)
    accuracy += (
        predicted_argmax == y.reshape((-1))
    ).sum()
    accuracy_pct = accuracy / (y.shape[0] * y.shape[1]) * 100
    print("Accuracy " , "sum loss", "count acc")
    print(accuracy_pct, sum_loss, count_full_win)
    
    with torch.no_grad():
        raw = torch.argmax(
            model(X[:1]).reshape((-1, config.vocab_size)),
            dim=1
        ).tolist()[::config.sequence_length]

        tokens, x_tokens = model.rollout(
            X[0].tolist(), 
            128,
            sampling="temperature"
        )
        print("Predicted vs actual tokens for X[0]")
        print("\t" + str(tokens[config.sequence_length:]))
        print("\t" + str(y[0].tolist()))
        print("Decoded (predicted vs actual)")
        decoded_tokens = source_vocab.decode(tokens)
        decoded_dataset = source_vocab.decode(X[0].tolist() + y[:1].tolist()[0])
        print("\t" + decoded_tokens)
        print("\t" + decoded_dataset)
        print("")

        if torch.all(predicted_argmax == y.reshape((-1))):
            count_full_win += 1
            for index, v in  enumerate(x_tokens):
                assert torch.all(X[index] == v), "Mismatch between tensor and input"
            assert decoded_dataset == decoded_tokens
