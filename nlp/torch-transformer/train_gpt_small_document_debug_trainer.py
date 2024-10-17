import torch
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset
from optimization_playground_shared.nlp.wordpiece.bpe import BPE
from optimization_playground_shared.nlp.wordpiece.bpeDocumentDecoder import get_document_dataset as get_document_dataset_bpe
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader

SEQUENCE_LENGTH = 8

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
    assert PADDING_IDX == 0

config = Config(
    vocab_size=source_vocab.size,
    embedding_dim=32,
    transformer_layers=2,
    attention_heads=4,
    dropout=0,
    feed_forward=128,
    padding_index=PADDING_IDX,
    sequence_length=SEQUENCE_LENGTH
)

def flatten_view(x, y):
    return x, y.view(-1)

def _accuracy_check(predicted, y, vocab: SimpleVocab):
    predicted = torch.argmax(
        predicted,
        dim=1
    )

    indices = torch.where(y != vocab.PADDING_IDX)
    y_filtered = y[indices]
    accuracy = (y[indices] == predicted[indices]).sum()

    return accuracy, y_filtered.shape[0]

model = GptTransformerModel(config)
epochs = 1024

optimi = torch.optim.Adam(
    model.parameters(),
    lr=13e-4
)

count_full_win = 0

trainer = TrainingLoop(
    model,
    optimi,
    loss=torch.nn.CrossEntropyLoss(ignore_index=PADDING_IDX),
    callback=flatten_view
)
trainer._accuracy_check = lambda x, y: _accuracy_check(x, y, source_vocab)

dataloader = get_dataloader(
    (X, y)
)

for _ in range(1024 * 8):
    model.train()
    optimi.zero_grad()

    loss, accuracy_pct = trainer.train(dataloader, callback=flatten_view)
    loss += loss

    print(accuracy_pct, loss, count_full_win)
    
    raw = torch.argmax(
        model(X[:1].to(trainer.device)).to(trainer.device).reshape((-1, config.vocab_size)),
        dim=1
    ).tolist()[::config.sequence_length]

    tokens, x_tokens = model.rollout(
        X[0].tolist(), 
        128,
        sampling="argmax"
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

    if accuracy_pct == 100:
        count_full_win += 1
        #for index, v in  enumerate(x_tokens):
        #    assert torch.all(X[index].to('cpu') == v.to('cpu')), "Mismatch between tensor and input"
        assert decoded_dataset == decoded_tokens
