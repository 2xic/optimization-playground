from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import torch
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics

from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.nlp.wordpiece.bpeDocumentDecoder import get_document_dataset
from torch.distributed.pipelining import SplitPoint
from pre_generator import get_bpe
from optimization_playground_shared.dataloaders.data_portal.Client import ZmqDataloader
from optimization_playground_shared.distributed.PipelineDistrubted import MultipleGpuBigModelWrapper

SEQUENCE_LENGTH = 512
batch_size = 128

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
metrics_tracker = Tracker("train_gpt_big_embeddings") if __name__ == "__main__" else None


def flatten_view(x, y):
    return x, y.view(-1)

class Trainer(MultipleGpuBigModelWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.clear_params()

    def clear_params(self):
        self.batch_count = 0
        self.batch_loss = 0
        self.batch_accuracy = 0

    def batch_done(self, losses, y: torch.Tensor, y_prediction: torch.Tensor):
      #  super().batch_done(losses, y, y_prediction)
        predicted_formatted = torch.argmax(y_prediction, dim=1)
        accuracy = ((y == predicted_formatted).sum())

        self.batch_loss += sum([i.item() for i in losses])
        self.batch_accuracy += accuracy
        self.batch_count += predicted_formatted.shape[0]

    def epoch_done(self, epoch):
        metrics_tracker.log(
            Metrics(
                epoch=epoch,
                loss=self.batch_loss,
                training_accuracy=self.batch_accuracy / self.batch_count * 100,
            )
        )
        self.clear_params()

def forward_dataloader(bpe, iterator, batch=32):
    X, y = get_document_dataset(bpe, [
        next(iterator) for _ in range(batch)
    ], SEQUENCE_LENGTH)
    raw_dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=batch_size,
        shuffle=True,
    )
    return raw_dataloader

def get_model():
    bpe = get_bpe()

    embedding_dim = 1024
    config = Config(
        vocab_size=len(bpe.index.tokens_index),
        embedding_dim=embedding_dim,
        transformer_layers=4,
        attention_heads=4,
        dropout=0.05,
        feed_forward=embedding_dim * 4,
        padding_index=bpe.get_system_token_index("<PADDING>"),
        sequence_length=SEQUENCE_LENGTH
    )
    model = GptTransformerModel(config)

    return bpe, model

def train_model():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    trainer = Trainer()
    trainer.start()

    bpe, model = get_model()

    dataloader = ZmqDataloader()
    iterator = iter(dataloader)

    raw_dataloader = forward_dataloader(
        bpe,
        iterator,
        batch=1,
    )
    trainer.setup(
        model,
        raw_dataloader,
        {
            "transformer_decoder.layers.1": SplitPoint.BEGINNING,
            "transformer_decoder.layers.2": SplitPoint.BEGINNING,
            "transformer_decoder.layers.3": SplitPoint.BEGINNING,
        },
    )
    trainer.load()
    for _ in range(100):
        raw_dataloader = forward_dataloader(
            bpe,
            iterator,
            batch=32
        )
        trainer.run_epoch(
            raw_dataloader,
            epochs=1,
            view_function=lambda x: x.view(-1)
        )
        # For each epoch we save the model
        trainer.save()    

if __name__ == "__main__":
    train_model()
