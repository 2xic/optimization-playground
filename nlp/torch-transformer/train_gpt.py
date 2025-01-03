from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.nlp.utils.sampling import argmax_sampling
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from torch.utils.data.distributed import DistributedSampler
from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import random
import json

SEQUENCE_LENGTH = 4
batch_size = 128
source_vocab = SimpleVocab()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
metrics_tracker = Tracker("train_gpt_shakespeare") if __name__ == "__main__" else None

def get_text_prediction(model: GptTransformerModel, seed: torch.Tensor):
    results = []
    raw_tokens = []
    with torch.no_grad():
        y = model.rollout(
            seed=seed,
            steps=32,
            sampling="argmax"
        )
        for index, i in enumerate(y):
            if index == seed.shape[-1]:
                results.append("*model output start*")
                raw_tokens.append(-42)
            results.append(source_vocab.vocab.index_vocab[i])
            raw_tokens.append(i)
    return " ".join(results), raw_tokens


def get_debug_prediction(model: GptTransformerModel, seed: torch.Tensor):
    results = []

    for i in seed.tolist():
        results.append(source_vocab.vocab.index_vocab[i])

    # print(seed.shape)
    with torch.no_grad():
        new_X = seed.clone()
        output = model(new_X.reshape((1, -1)).to(model.device))
        for i in range(output.shape[0]):
            pred_i = output[i, :].reshape((1, -1))
            index = argmax_sampling(
                pred_i
            )
            results.append(
                str((str(i), source_vocab.vocab.index_vocab[index.item()]))
            )

    return " ".join(results)

def get_dataloader():
    text = None
#    with open("shakespeare.txt", "r") as file:
#        text = file.read()
    text = """
    I love bagels, this is just a test to make sure the model is doing something reasonable. How reasonable is all of this ?


    I think the model should learn this text when overfitting.
    """

    X, y = get_document_dataset(source_vocab, [text], SEQUENCE_LENGTH)
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
        debug_x = X[random.randint(0, X.shape[0] - 1)]
        text, raw_tokens = get_text_prediction(model, debug_x)
        debug_text = get_debug_prediction(model, debug_x)
        metrics_tracker._log(
            Metrics(
                epoch=epoch,
                loss=loss,
                training_accuracy=accuracy,
                prediction=Prediction.text_prediction(
                    "\n".join([
                        "text: ",
                        text,
                        "tokens: ",
                        json.dumps(raw_tokens),
                        "debug_text:",
                        debug_text
                    ])
                )
            )
        )
        scheduler.step()

if __name__ == "__main__":
    train_model()
