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
from optimization_playground_shared.nlp.wordpiece.bpe import BPE

SEQUENCE_LENGTH = 32
batch_size = 32
preload = False
bpe = get_bpe() if preload else BPE()

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

    def clear_params(self):
        self.batch_count = 0
        self.batch_loss = 0
        self.batch_accuracy = 0

    def batch_done(self, losses, X: torch.Tensor, y: torch.Tensor, y_prediction: torch.Tensor):
        predicted_formatted = torch.argmax(y_prediction, dim=1)

        indices = torch.where(y != self.ignore_index)
        y_filtered = y[indices]
        accuracy = ((y[indices] == predicted_formatted[indices]).sum())

        self.batch_loss += sum([i.item() for i in losses])
        self.batch_accuracy += accuracy
        self.batch_count += y_filtered.shape[0]

    def epoch_done(self, epoch, is_last_stage):
        # only include sometimes, not always
        include_output = epoch % 5 == 0

        # if it is the last stage then output it.
        input_tensor_temperature = self.get_model_prediction() if include_output else None

        if is_last_stage:
            output = Prediction.text_prediction(
                        "\n".join([
                            "input:",
                            bpe.decode(self.batch_X[0].tolist()[:32]),
                            "\n\n",
                            "expected:",
                            bpe.decode(self.batch_y[0].tolist()[:32]),
                            "\n\n",
                            "temperature:",
                            bpe.decode(input_tensor_temperature)[:1024],
                            "\n\n",
                        ])  
                    ) if include_output else None
            metrics_tracker.queue(
                Metrics(
                    epoch=epoch,
                    loss=self.batch_loss,
                    training_accuracy=self.batch_accuracy / self.batch_count * 100,
                    prediction=output,
                )
            )
            self.clear_params()

    def get_model_prediction(self):
        print("entering ", self.rank)
        dist.barrier()
        self.module.eval()
        # already done per stage ... in the step function
        # torch.cuda.empty_cache()   
        # Only last stage is useful
        old_microbatches = self.schedule._n_microbatches
        self.schedule._n_microbatches = 1
        input_tensor_temperature = self.get_predictions(
            bpe.decode(self.batch_X[0].tolist()[:32]),
            sample="temperature"
        )
        dist.barrier()
        
        # re-enable training
        self.schedule._n_microbatches = old_microbatches
        self.module.train()

        return input_tensor_temperature

    def get_predictions(self, seed_text, sample="temperature", steps=64):
        input_tensor = bpe.encode(
            seed_text
        )
        with torch.no_grad():
            for _ in tqdm(range(steps), desc="predictions"):
                next_input_tensor = torch.zeros((1)).long().to(self.device)
                output = self.small_rollout(input_tensor)
                if output is not None:
                    normalized_input = output
                    if sample == "temperature":
                        input_tensor.append(temperature_sampling(normalized_input).item())
                    else:
                        input_tensor.append(argmax_sampling(normalized_input).item())
                    next_input_tensor[0] = torch.tensor(input_tensor[-1])
                    for rank in range(self.world_size - 1):
                        dist.send(tensor=next_input_tensor, dst=rank)
                if output is None:
                    dist.recv(tensor=next_input_tensor, src=self.world_size - 1)
                    input_tensor.append(next_input_tensor.item())
                dist.barrier()
        return input_tensor

    def small_rollout(self, X):
        X_tensor = torch.full((batch_size // self.world_size, SEQUENCE_LENGTH), bpe.index.padding_idx, device=self.device)
        X_tensor[:, :len(X)] = torch.tensor(X[-SEQUENCE_LENGTH:])
        return self.predict(X_tensor)

    def predict(self, X):
        results = self.forward(X)
        if results is None:
            return
        results = results.reshape((results.shape[0] // SEQUENCE_LENGTH, SEQUENCE_LENGTH, bpe.size))
        return results[0, ::SEQUENCE_LENGTH, :].reshape((1, -1))

def forward_dataloader(bpe: BPE, zmq: ZmqDataloader, batch=32):
    # sequence is the previously loaded data + a new batch so the dataset is always increasing.
    sequence = [
        zmq[index] for index in range(min(batch, zmq.__len__()))
    ]
#    sequence = [
#        "hello world, this is just some random text to verify if the model can learn something."
#    ]
    if not preload and not bpe.index.is_readonly:
        for content in sequence:
            bpe.add_vocab(content)
        bpe.merge(
            n=100
        )

    X, y = get_document_dataset(bpe, sequence, SEQUENCE_LENGTH)
    raw_dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=batch_size,
        shuffle=True, 
        drop_last=True
    )
    assert X.shape[0] > 0
    return raw_dataloader

def get_model():
    embedding_dim = 128
    config = Config(
        vocab_size=bpe.size,
        embedding_dim=embedding_dim,
        transformer_layers=4,
        attention_heads=4,
        dropout=0,
        feed_forward=embedding_dim * 4,
        padding_index=bpe.index.padding_idx,
        sequence_length=SEQUENCE_LENGTH
    )
    model = GptTransformerModel(config)

    return model

def start_dataloader_background_thread(data_queue: queue.Queue, bpe, dataloader: ZmqDataloader):
    # Load in one new batch of data 
    raw_dataloader = forward_dataloader(
        bpe,
        dataloader,
        batch=32
    )
    data_queue.put(raw_dataloader)

def train_model():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # dataloader setup
    dataloader = ZmqDataloader()
    dataloader.max_document_size = 1
    dataloader_queue = queue.Queue(maxsize=1)
    raw_dataloader = forward_dataloader(
        bpe,
        dataloader,
        batch=1,
    )

#    print(f"Size {bpe.size}")
#    print(f"readonly == {bpe.index.is_readonly}")

    bpe.index.is_readonly = True

    # Model setup
    model = get_model()
    trainer = Trainer(
        padding_index=bpe.index.padding_idx
    )
    trainer.start()

    Thread(target=start_dataloader_background_thread, args=(dataloader_queue, bpe, dataloader, )).start()    

    trainer.setup(
        model,
        raw_dataloader,
        {
            "transformer_decoder.layers.1": SplitPoint.BEGINNING,
            "transformer_decoder.layers.2": SplitPoint.BEGINNING,
            "transformer_decoder.layers.3": SplitPoint.BEGINNING,
        },
    )

    bpe.lock()

#    Thread(target=start_dataloader_background_thread, args=(dataloader_queue, bpe, dataloader, )).start()
    raw_dataloader = dataloader_queue.get()
    for epoch in range(1024):
        # NOTE: any thread calls here will be slow so do it with one background thread, don't start many.
        trainer.run_epoch(
            raw_dataloader,
            epochs=1,
            view_function=lambda x: x.view(-1)
        )
        if epoch % 10 == 0:
            # For each epoch we save the model
            trainer.save()    
    print("Done")
    # trainer.destroy()
    metrics_tracker.stop()

if __name__ == "__main__":
    train_model()
