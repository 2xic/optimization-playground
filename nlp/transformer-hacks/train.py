from model import Model, Config, DEVICE, TransformerLayerType
from transformer_dataset import TransformerDatasetBase, TransformerTextDataset
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
)
from typing import Callable, Tuple, Optional
from dataset_tokenizer import SimpleTextEncoder
from tqdm import tqdm
from dataclasses import dataclass
import os
import time
from performance_benchmarker import Timer
from torch.optim.lr_scheduler import LambdaLR

DEBUG = False
# This slows down the training a lot ...
VALIDATION_DEBUG = False

BETA_1 = 0.90
BETA_2 = 0.95


class ModelStateSaver:
    def __init__(self, name):
        self.name = name

    def save(self, model, optimizer, epoch, loss):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "epoch": epoch,
            "loss": loss,
        }
        torch.save(checkpoint, self.get_file_path())
        torch.save(checkpoint, self.get_file_path_epoch(epoch))
        print("save checkpoint")

    def load_model_state(self, model: torch.nn.Module):
        state = torch.load(self.get_file_path())
        model.load_state_dict(state["model_state_dict"])
        return model

    def get_file_path(self):
        dir_name = os.path.join(os.path.dirname(__file__), self.name)
        os.makedirs(dir_name, exist_ok=True)
        return os.path.join(dir_name, "checkpoint.pth")

    def get_file_path_epoch(self, epoch):
        dir_name = os.path.join(os.path.dirname(__file__), self.name)
        os.makedirs(dir_name, exist_ok=True)
        return os.path.join(dir_name, f"{epoch}_checkpoint.pth")


class TrainingTimer:
    def __init__(self, minutes):
        self.start = time.time()
        self.minutes = minutes

    def done(self):
        return (time.time() - self.start) > self.minutes * 60

    def reset(self):
        self.start = time.time()


@dataclass
class TrainingOptions:
    batch_size: int = 32 if DEVICE.type != "cuda" else 256
    epochs: int = 100


def debug_print(*args):
    if DEBUG:
        print(*args)


def create_config(vocab_size, padding_index, sequence_length):
    return Config(
        sequence_length=sequence_length,
        dim_embeddings=32,
        num_attention_heads=4,
        num_transformer_layers=4,
        padding_index=padding_index,
        vocab_size=vocab_size,
        transformer_layer=TransformerLayerType.GPT2,
    )


def timer_iterator(dataset):
    with Timer("create_iterator"):
        iterator = iter(dataset)
    for i in range(len(dataset)):
        with Timer("timer_iterator"):
            X, y = next(iterator)
        yield X, y


def train(
    dataset: TransformerDatasetBase,
    override: Callable[[Config], Config] = (lambda x: x),
    create_model: Callable[[Config], Model] = (lambda x: Model(x)),
    create_optimizer: Callable[
        [torch.nn.Module, Config], Tuple[optim.Optimizer, Optional[LambdaLR]]
    ] = lambda model, config: (
        optim.Adam(model.parameters(), lr=config.learning_rate, betas=(BETA_1, BETA_2)),
        None,
    ),
    options: TrainingOptions = TrainingOptions(),
    progress=lambda x: x,
    sampling=temperature_sampling,
    checkpoint=False,
):
    config = create_config(
        dataset.vocab_size,
        dataset.padding_index,
        dataset.sequence_size,
    )
    config = override(config)
    model = create_model(config).to(DEVICE)
    optimizer, scheduler = create_optimizer(model, config)

    state_saver = (
        ModelStateSaver(config.model_name) if config.model_name is not None else None
    )
    loader = dataset.iter(batch_size=options.batch_size)

    epochs = []
    epochs_loss = []
    epochs_accuracy = []
    for epoch in range(options.epochs):
        timer = TrainingTimer(minutes=30)
        sum_loss = 0
        accuracy = 0
        rows = 0
        with Timer("epoch"):
            with Timer("creating_iterator"):
                tqdm_loader = progress(loader)
                iterator = timer_iterator(tqdm_loader)
            for index, (X, y) in enumerate(iterator):
                with Timer("move_to_device"):
                    X, y = X.to(DEVICE), y.to(DEVICE)
                with Timer("predictions"):
                    y_predicted = model(X)

                with Timer("optimizer"):
                    loss = torch.nn.functional.cross_entropy(
                        y_predicted.view(-1, config.vocab_size),
                        y.view(-1),
                        ignore_index=config.padding_index,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    if config.max_grad_norm != None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.max_grad_norm
                        )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    sum_loss += loss.item()
                    # Accuracy metrics
                    y_sample_next = sampling(y_predicted[:, -1, :])
                    y_next = y[:, -1]
                    assert y_sample_next.shape == y_next.shape
                    accuracy += (y_sample_next == y_next).sum()
                    rows += y_next.shape.numel()

                if (
                    VALIDATION_DEBUG
                    and index % 10 == 0
                    and isinstance(tqdm_loader, tqdm)
                ):
                    with Timer("update stats"):
                        tqdm_loader.set_description(
                            "Epoch {epoch}, Accuracy {acc}, Loss {loss}".format(
                                epoch=epoch, acc=(accuracy / rows * 100), loss=sum_loss
                            )
                        )

                if checkpoint and timer.done() and state_saver is not None:
                    with Timer("checkpoints"):
                        state_saver.save(model, optimizer, epoch, sum_loss)
                        timer.reset()

        if checkpoint and state_saver is not None:
            state_saver.save(model, optimizer, epoch, sum_loss)
            timer.reset()

        acc = accuracy / rows * 100
        epochs_accuracy.append(acc.item())
        epochs_loss.append(loss.item())
        epochs.append(epoch)

        if VALIDATION_DEBUG:
            assert acc <= 100, acc
            debug_print(f"epoch: {epoch}, loss: {sum_loss}, accuracy {acc}")

            with torch.no_grad():
                rows = dataset.sample(n=2)
                for i, j in rows:
                    i, j = i.to(DEVICE), j.to(DEVICE)
                    i = i.reshape((1, -1))
                    predicted = model(i)[0]
                    word_idx = sampling(predicted)

                    next_word_idx = word_idx[-1].item()
                    expected_word_idx = j[-1].item()

                    input_document = i[0]
                    context = "".join(dataset.decode_tokens(input_document.tolist()))
                    debug_print(f"\tcontext: {context}")

                    word = dataset.decode_tokens([next_word_idx])
                    expected = dataset.decode_tokens([expected_word_idx])
                    debug_print(f"\tnext token: '{word}'")
                    debug_print(f"\texpected token: '{expected}'")
                    debug_print("")
            # Check that the model converges to something.
            with torch.no_grad():
                accuracy = 0
                for index, (X, y) in enumerate(dataset.sample(128)):
                    X = X.to(DEVICE)
                    predicted = model(X.reshape((1, -1)))
                    y_sample_next = sampling(predicted[:, -1, :])
                    accuracy += y_sample_next.item() == y[-1].item()
                debug_print((accuracy / index) * 100)

    return (epochs, epochs_accuracy, epochs_loss, model)


if __name__ == "__main__":
    tokenizer, cached = SimpleTextEncoder("example").load_cache()
    if not cached:
        print("Not cached building tokenizer.")
        tokenizer = tokenizer.build_from_files(["example.text"])
        tokenizer.save_cache()
    else:
        print("Tokenizer is cached.")
    text_dataset = TransformerTextDataset.from_file(
        tokenizer, "example.text", sequence_length=4
    )
    train(
        text_dataset,
        options=TrainingOptions(batch_size=256),
        progress=lambda x: tqdm(range(x)),
    )
