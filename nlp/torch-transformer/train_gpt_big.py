from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
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
from torch.distributed.pipelining import pipeline, SplitPoint, Pipe

SEQUENCE_LENGTH = 1024
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
            steps=512,
            sampling="temperature"
        )
        for index, i in enumerate(y):
            if index == seed.shape[-1]:
                results.append("*model output start*")
                raw_tokens.append(-42)
            results.append(source_vocab.vocab.index_vocab[i])
            raw_tokens.append(i)
    return " ".join(results), raw_tokens

def get_dataloader():
    text = None
    with open("shakespeare.txt", "r") as file:
        text = file.read()

    X, y = get_document_dataset(source_vocab, [
        "\n".join(
            text.split("\n")[:10]
        )
    ], SEQUENCE_LENGTH)
    source_vocab.lock()

    return X, y

def flatten_view(x, y):
    return x, y.view(-1)

from optimization_playground_shared.distributed.PipelineDistrubted import MultipleGpuBigModelWrapper


class Trainer(MultipleGpuBigModelWrapper):
    def __init__(self) -> None:
        super().__init__()

    def batch_done(self, losses, y: torch.Tensor, y_prediction: torch.Tensor):
        super().batch_done(losses, y, y_prediction)
        predicted_formatted = torch.argmax(y_prediction, dim=1)
        accuracy = ((y == predicted_formatted).sum() / predicted_formatted.shape[0] * 100)
        print(f"Accuracy {accuracy}")

def train_model():
    trainer = Trainer()
    trainer.start()

    X, y = get_dataloader()

    assert X.shape[0] == y.shape[0]

    embedding_dim = 256
    config = Config(
        vocab_size=source_vocab.size,
        embedding_dim=embedding_dim,
        transformer_layers=2,
        attention_heads=4,
        dropout=0.05,
        feed_forward=embedding_dim * 4,
        padding_index=source_vocab.vocab.PADDING_IDX,
        sequence_length=SEQUENCE_LENGTH
    )

    dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=batch_size,
        shuffle=False,
    )


    model = GptTransformerModel(config)
    print(model)
    trainer.run(
        model,
        dataloader,
        {
         #   "embedding": SplitPoint.BEGINNING,
            "transformer_decoder": SplitPoint.BEGINNING,
        #    "transformer_decoder.layers.0": SplitPoint.BEGINNING,
        #    "transformer_decoder.layers.1": SplitPoint.BEGINNING,
        #    "transformer_decoder.layers.2": SplitPoint.BEGINNING,
        #    "transformer_decoder.layers.3": SplitPoint.BEGINNING,
    #            "layer_norm": SplitPoint.BEGINNING,
        },
        view_function=lambda x: x.view(-1)
    )

if __name__ == "__main__":
    train_model()
