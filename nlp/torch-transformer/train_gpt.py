from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from torch.utils.data.distributed import DistributedSampler
from optimization_playground_shared.nlp.SimpleVocab import splitter
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import random
import json

SEQUENCE_LENGTH = 64
batch_size = 128
source_vocab = SimpleVocab()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
metrics_tracker = Tracker("train_gpt_shakespeare") if __name__ == "__main__" else None

def get_document_words(text):
    # todo: create a good tokenizer, this does not really work for code tokens
    if type(text) == bytes:
        return splitter(text.decode())
    else:
        return splitter(text)

def encode_document_text(vocab: SimpleVocab, text, tensor_x, tensor_y, entries_index):
    words = get_document_words(text)
    count_words = len(words)
    vocab_add = vocab.vocab.add 
    # preload all the words fast
    words = torch.tensor(list(map(lambda x: vocab_add(x), words)), dtype=torch.long)
    for i in range(1, count_words - 1):
        start_index = 0
        if i > SEQUENCE_LENGTH:
            start_index = i - SEQUENCE_LENGTH
        context = words[start_index:i]
        next_token = words[i+1:i+SEQUENCE_LENGTH + 1]
        # add the entries
        tensor_x[entries_index, :context.shape[-1]] = context
        tensor_y[entries_index, :next_token.shape[-1]] = next_token
        entries_index += 1
    return tensor_x, tensor_y, entries_index

def get_document_dataset(vocab: SimpleVocab, documents):
    assert type(documents) == list
    entries_count = 0
    for i in documents:
        # -2 as we need the first token to be placed and the last token
        entries_count += len(get_document_words(i)) - 2

    X = torch.full(size=(entries_count, SEQUENCE_LENGTH), fill_value=vocab.vocab.PADDING_IDX, dtype=torch.long)
    y = torch.full(size=(entries_count, SEQUENCE_LENGTH), fill_value=vocab.vocab.PADDING_IDX, dtype=torch.long)
    entries_index = 0
    for document in documents:
        X, y, entries_index = encode_document_text(vocab, document, X, y, entries_index)
  #  print(torch.bincount(y).tolist())#, y.unique(sorted=True).float()))
    assert not torch.all(X == 0), "All zeros is bad"
    assert not torch.all(y == 0), "All zeros is bad"

    # Random sampling out of the dataset for better coverage
    indices = torch.randint(0, X.size(0), (entries_count // 32,))
    return X[indices], y[indices]


def get_text_prediction(model: GptTransformerModel, seed: torch.Tensor):
    results = []
    raw_tokens = []
#    seed = source_vocab.get_tensor("second citizen : ", sequence_length=-1).reshape(-1)
    with torch.no_grad():
        y = model.rollout(
            seed=seed,
            steps=512,
            sampling="temperature"
        )
        for index, i in enumerate(y):
            if index == seed.shape[-1]:
                results.append("*model output start*")
                raw_tokens.append(-42)
            #    print((index, seed.shape[-1]))
            #    print(results[-1])
            results.append(source_vocab.vocab.index_vocab[i])
            raw_tokens.append(i)
    return " ".join(results), raw_tokens

def get_dataloader():
    text = None
    with open("shakespeare.txt", "r") as file:
        text = file.read()

    X, y = get_document_dataset(source_vocab, [text])
    print(X)
    print(y)
    source_vocab.lock()

    return X, y

def flatten_view(x, y):
    return x, y.view(-1)

def train_model():
    X, y = get_dataloader()
    embedding_dim = 256
    config = Config(
        vocab_size=source_vocab.size,
        embedding_dim=embedding_dim,
        transformer_layers=4,
        attention_heads=4,
        dropout=0.05,
        feed_forward=embedding_dim * 4,
        padding_index=source_vocab.vocab.PADDING_IDX,
        sequence_length=SEQUENCE_LENGTH
    )
    model = GptTransformerModel(config)
    epochs = 1024
    optimizer = optim.Adam(model.parameters(), lr=0.00004, weight_decay=0.01)

    dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=batch_size,
        shuffle=False,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs)
    trainer = TrainingLoop(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=source_vocab.vocab.PADDING_IDX))
    for epoch in range(epochs):
        (loss, accuracy) = trainer.use_tqdm().train(
            dataloader,
            callback=flatten_view,
        )
        text, raw_tokens = get_text_prediction(model, X[random.randint(0, X.shape[0] - 1)])
        metrics_tracker.log(
            Metrics(
                epoch=epoch,
                loss=loss,
                training_accuracy=accuracy,
                prediction=Prediction.text_prediction(
                    "\n".join([
                        "text: ",
                        text,
                        "tokens: ",
                        json.dumps(raw_tokens)
                    ])
                )
            )
        )
        scheduler.step()

if __name__ == "__main__":
    train_model()