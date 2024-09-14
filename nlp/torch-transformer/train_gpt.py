from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.distributed.TrainWrapper import MultipleGpuTrainWrapper
from torch.utils.data.distributed import DistributedSampler
from optimization_playground_shared.nlp.SimpleVocab import splitter
from optimization_playground_shared.training_loops.TrainingLoopAccumulate import TrainingLoopAccumulate

SEQUENCE_LENGTH = 128
batch_size = 32
source_vocab = SimpleVocab()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
metrics_tracker = Tracker("train_gpt_shakespare")

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


def get_text_prediction(model: GptTransformerModel):
    results = []
    seed = source_vocab.get_tensor("second citizen : ", sequence_length=-1).reshape(-1)
    with torch.no_grad():
        y = model.rollout(
            seed=seed,
            steps=512,
        )
        for i in y:
            results.append(source_vocab.vocab.index_vocab[i])
    return " ".join(results)

def get_dataloader():
    text = None
    with open("shakespeare.txt", "r") as file:
        text = file.read()

    X, y = get_document_dataset(source_vocab, [text])
    source_vocab.lock()

    return X, y

def flatten_view(x, y):
    return x, y.view(-1)

def train_model():
    X, y = get_dataloader()

    config = Config(
        vocab_size=source_vocab.size,
        embedding_dim=128,
        transformer_layers=2,
        attention_heads=4,
        dropout=0.1,
        feed_forward=128,
        padding_index=source_vocab.vocab.PADDING_IDX,
        sequence_length=SEQUENCE_LENGTH
    )
    model = GptTransformerModel(config)
    optimizer = optim.Adam(model.parameters())    
    dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=batch_size,
        shuffle=False,
    )
    trainer = TrainingLoopAccumulate(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=source_vocab.vocab.PADDING_IDX))
    for epoch in range(128):
        (loss, accuracy) = trainer.use_tqdm().train(
            dataloader,
            callback=flatten_view,
        )
        metrics_tracker.log(
            Metrics(
                epoch=epoch,
                loss=loss,
                training_accuracy=accuracy,
                prediction=Prediction.text_prediction(
                    get_text_prediction(model))
            )
        )

if __name__ == "__main__":
    train_model()