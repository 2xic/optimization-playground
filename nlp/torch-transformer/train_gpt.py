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

SEQUENCE_LENGTH = 128
batch_size = 32
source_vocab = SimpleVocab()
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
metrics_tracker = Tracker("train_gpt")

print("Starting .... ")


def sliding_vocab(text):
    X = []
    y = []

    words = text.lower().replace(":", " : ").strip().split(" ")
    words = list(filter(lambda x: len(x), words))
    for i in range(1, len(words) - 1):
        X.append(source_vocab.get_tensor(
            " ".join(words[max(i-SEQUENCE_LENGTH, 0):i]), sequence_length=SEQUENCE_LENGTH))
        y.append(source_vocab.get_tensor(
            " ".join(words[i:i+1]), sequence_length=1)[0])
        if 32 * 2048 < len(X):
            break
    return torch.concat(X), torch.concat(y)


def get_dataloader():
    text = None
    with open("tinyshakespeare.txt", "r") as file:
        text = file.read()

    X, y = sliding_vocab(text)
    source_vocab.lock()
    return X, y


def get_text_prediction(model, device):
    results = []
    seed = source_vocab.get_tensor("second citizen : ", sequence_length=-1).reshape(-1)
    print(seed)
    with torch.no_grad():
        y = model.rollout(
            seed=seed,
            steps=512,
            device=device,
        )
        for i in y:
            results.append(source_vocab.vocab.index_vocab[i])
    return " ".join(results)


class Trainer(MultipleGpuTrainWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.training_loss = []
        self.training_accuracy = []

    def _get_model_and_optimizer(self):
        assert 10 < source_vocab.size, "You failed to initialize the vocab ?"
        config = Config(
            vocab_size=source_vocab.size,
            embedding_dim=128,
            transformer_layers=2,
            attention_heads=4,
            dropout=0.1,
            feed_forward=128,
            padding_index=source_vocab.vocab.PADDING_IDX,
            sequence_size=SEQUENCE_LENGTH
        )
        model = GptTransformerModel(config)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer

    def _get_dataloader(self, device, is_debug_mode):
        X, y = get_dataloader()
        sampler = None
        if not is_debug_mode:
            def sampler(dataset): return DistributedSampler(
                dataset, shuffle=True)
        return get_raw_dataloader((
            X.clone(),
            y.clone()
        ),
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler
        )

    def _epoch_done(self, epoch, model, loss, accuracy, device):
        metrics_tracker.log(
            Metrics(
                epoch=epoch,
                loss=loss,
                training_accuracy=accuracy,
                prediction=Prediction.text_prediction(
                    get_text_prediction(model, device))
            )
        )
        results = []
        seed = source_vocab.get_tensor(
            "second citizen : ", sequence_length=-1).reshape(-1)

        with torch.no_grad():
            y = model.rollout(
                seed=seed,
                steps=512,
                device=device,
            )
            for i in y:
                results.append(source_vocab.vocab.index_vocab[i])
        return " ".join(results)

    def _loss(self):
        return torch.nn.CrossEntropyLoss(ignore_index=source_vocab.vocab.PADDING_IDX)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()
