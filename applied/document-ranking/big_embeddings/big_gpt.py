from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling, argmax_sampling
import torch
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.metrics_tracker.producer import Tracker
from optimization_playground_shared.metrics_tracker.metrics import Metrics, Prediction

from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.nlp.wordpiece.bpeDocumentDecoder import get_document_dataset
from torch.distributed.pipelining import SplitPoint
from .pre_generator import get_bpe
from optimization_playground_shared.dataloaders.data_portal.Client import ZmqDataloader
from optimization_playground_shared.distributed.PipelineDistrubted import MultipleGpuBigModelWrapper
from threading import Thread
import queue
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

SEQUENCE_LENGTH = 512
batch_size = 128
bpe = get_bpe()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
metrics_tracker = Tracker("train_gpt_big_embeddings") if __name__ == "__main__" else None


def flatten_view(x, y):
    return x, y.view(-1)

class Trainer(MultipleGpuBigModelWrapper):
    def __init__(self, padding_index) -> None:
        self.ignore_index = padding_index
        super().__init__(
            loss_function=torch.nn.CrossEntropyLoss(
                ignore_index=self.ignore_index
            )
        )
        self.clear_params()

        self.batch = None

    def clear_params(self):
        self.batch_count = 0
        self.batch_loss = 0
        self.batch_accuracy = 0

    def batch_done(self, losses, X: torch.Tensor, y: torch.Tensor, y_prediction: torch.Tensor):
      #  super().batch_done(losses, y, y_prediction)
        predicted_formatted = torch.argmax(y_prediction, dim=1)
        print("y_prediction ", y_prediction.shape)

        indices = torch.where(y != self.ignore_index)
        y_filtered = y[indices]
        accuracy = ((y[indices] == predicted_formatted[indices]).sum())

        """
        indices = torch.randint(0, y.size(0), size=(32,)).long()
        print(indices, y.shape)
        self.batch = (
            X[indices].detach(),
            y[indices].detach(),
            torch.argmax(y_prediction[indices].detach(), dim=1)
        )
        """
        self.batch_loss += sum([i.item() for i in losses])
        self.batch_accuracy += accuracy
        self.batch_count += y_filtered.shape[0]

    def epoch_done(self, epoch, is_last_stage):
        dist.barrier()
        # Only last stage is useful
        old_microbatches = self.schedule._n_microbatches
        self.schedule._n_microbatches = 1
        input_tensor_temperature = self.single_rollout(
            "hello world"
        )
        input_tensor_argmax = self.single_rollout(
            "hello world",
            sample="argmax"
        )
        dist.barrier()
        self.schedule._n_microbatches = old_microbatches
        if is_last_stage:
            output = Prediction.text_prediction(
                        "\n".join([
                            "temperature:\n",
                            bpe.decode(input_tensor_temperature)[:1024],
                            "\n\n",
                            "argmax:\n",
                            bpe.decode(input_tensor_argmax)[:1024],
                        ])  
                    )
            metrics_tracker.log(
                Metrics(
                    epoch=epoch,
                    loss=self.batch_loss,
                    training_accuracy=self.batch_accuracy / self.batch_count * 100,
                    prediction=output,
                )
            )
            self.clear_params()

    def get_predictions(self, seed_text, sample="temperature"):
        input_tensor = bpe.encode_sentences(
            seed_text
        )
        with torch.no_grad():
            for _ in tqdm(range(1024), desc="predictions"):
                next_input_tensor = torch.zeros((1)).long().to(self.device)
                output = self.small_rollout(input_tensor)
                if output is not None:
                    if sample == "temperature":
                        input_tensor.append(temperature_sampling(F.softmax(output, dim=1)).item())
                    else:
                        input_tensor.append(argmax_sampling(F.softmax(output, dim=1)).item())
                    next_input_tensor[0] = torch.tensor(input_tensor[-1])
                    for rank in range(self.world_size - 1):
                        dist.send(tensor=next_input_tensor, dst=rank)
                if output is None:
                    dist.recv(tensor=next_input_tensor, src=self.world_size - 1)
                    input_tensor.append(next_input_tensor.item())
                dist.barrier()
        return input_tensor

    def small_rollout(self, X):
        # 32 also works ? 
        X_tensor = torch.full((32, SEQUENCE_LENGTH), bpe.get_system_token_index("<PADDING>")).to(self.device)
        X_tensor[:, :len(X)] = torch.tensor(X[-SEQUENCE_LENGTH:])
#        assert X_tensor.shape[0] > 1, X_tensor.shape[0] 
        return self.predict(X_tensor)

    def single_rollout(self, seed_text, sample="temperature"):
        X = bpe.encode_sentences(
            seed_text
        )
        # 32 also works ? 
        X_tensor = torch.full((32, SEQUENCE_LENGTH), bpe.get_system_token_index("<PADDING>")).to(self.device)
        X_tensor[:, :len(X)] = torch.tensor(X[-SEQUENCE_LENGTH:])
#        assert X_tensor.shape[0] > 1, X_tensor.shape[0] 
        output = self.forward(X_tensor)
        if output is not None:
            for index in range(output.shape[0]):
                output_t = output[index].reshape((1, -1))
                if sample == "temperature":
                    X.append(temperature_sampling(F.softmax(output_t, dim=1)).item())
                else:
                    X.append(argmax_sampling(F.softmax(output_t, dim=1)).item())
        return X

    def predict(self, X, next_token_location=-1):
        results = self.forward(X)
        if results is None:
            return
        if next_token_location == -1:
            next_token_location = results.shape[0] -1
        return results[next_token_location, :].reshape((1, -1))

def forward_dataloader(bpe, iterator, zmq: ZmqDataloader, batch=32):
    # sequence is the previously loaded data + a new batch so the dataset is always increasing.
    sequence = [
        "hello world, I love bagels. Maybe if I try to overfit the model then this thing will work."
        for _ in range(batch_size)
    ]
    X, y = get_document_dataset(bpe, sequence, SEQUENCE_LENGTH)
    raw_dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=batch_size,
        shuffle=True,
    )
    assert X.shape[0] > 0
    return raw_dataloader

def get_model():
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

def start_dataloader_background_thread(data_queue: queue.Queue, bpe, iterator, dataloader: ZmqDataloader, batch=32):
    # Load in one new batch of data 
    raw_dataloader = forward_dataloader(
        bpe,
        iterator,
        dataloader,
        batch=32
    )
    data_queue.put(raw_dataloader)

def train_model():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    bpe, model = get_model()

    trainer = Trainer(
        padding_index=bpe.get_system_token_index("<PADDING>")
    )
    trainer.start()

    dataloader = ZmqDataloader()
    dataloader.max_document_size = 1
    print(len(dataloader))
    iterator = iter(dataloader)

    dataloader_queue = queue.Queue(maxsize=1)
    raw_dataloader = forward_dataloader(
        bpe,
        iterator,
        dataloader,
        batch=1,
    )

    Thread(target=start_dataloader_background_thread, args=(dataloader_queue, bpe, iterator, dataloader, )).start()    

    trainer.setup(
        model,
        raw_dataloader,
        {
            "transformer_decoder.layers.1": SplitPoint.BEGINNING,
            "transformer_decoder.layers.2": SplitPoint.BEGINNING,
            "transformer_decoder.layers.3": SplitPoint.BEGINNING,
        },
    )
#    trainer.load()

    for _ in range(100):
        raw_dataloader = dataloader_queue.get()
        Thread(target=start_dataloader_background_thread, args=(dataloader_queue, bpe, iterator, dataloader, )).start()
        trainer.run_epoch(
            raw_dataloader,
            epochs=1,
            view_function=lambda x: x.view(-1)
        )
        # For each epoch we save the model
        trainer.save()    

if __name__ == "__main__":
    train_model()
