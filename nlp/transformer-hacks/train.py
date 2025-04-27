from model import Model, Config, DEVICE, TransformerLayerType
from transformer_dataset import TransformerDatasetBase, TransformerTextDataset
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
)
from typing import Callable
from dataset_tokenizer import SimpleTextEncoder
from tqdm import tqdm
from dataclasses import dataclass
import os 
import time
# FOR DEBUGGING ONLY
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config as GtpConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import get_cosine_schedule_with_warmup

DEBUG = False

print(DEVICE)

class ModelStateSaver:
    def __init__(self, name):
        self.name = name

    def save(self, model, optimizer, epoch, loss):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, self.get_file_path())

    def load_model_state(self, model: torch.nn.Module):
        state = torch.load(self.get_file_path())
        model.load_state_dict(
            state["model_state_dict"]
        )
        return model

    def get_file_path(self):
        dir_name = os.path.join(
            os.path.dirname(__file__),
            self.name
        )
        os.makedirs(dir_name, exist_ok=True)
        return os.path.join(
            dir_name,
            "checkpoint.pth"
        )

class TrainingTimer:
    def __init__(self, minutes):
        self.start = time.time()
        self.minutes = minutes

    def done(self):
        return (time.time() - self.start) > self.minutes * 60


@dataclass
class TrainingOptions:
    batch_size: int = (32 if DEVICE.type != "cuda" else 256)
    learning_rate: float = 3e-4
    epochs: int = 100
    max_grad_norm: float = 1

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


def train(
    dataset: TransformerDatasetBase, 
    override: Callable[[Config], Config] = (lambda x: x),
    create_model: Callable[[Config], Model] = (lambda x: Model(x)),
    options: TrainingOptions = TrainingOptions(),
    progress=range
):
    config = create_config(
        dataset.vocab_size,
        dataset.padding_index,
        dataset.sequence_size,
    )
    config = override(config)
    model = create_model(config).to(DEVICE)
    if False:
        model = GptTransformerModel(GtpConfig(
            config.vocab_size,
            config.dim_embeddings,
            config.sequence_length,
            config.num_transformer_layers,
            config.num_attention_heads,
            config.dropout,
            128,
            config.padding_index,
        )).to(DEVICE)
        model.forward = model.raw_forward


    state_saver = ModelStateSaver("loading-test")
    optimizer = optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=1e-2)
    loader = dataset.iter(batch_size=options.batch_size)

    total_steps = len(loader) * options.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(total_steps * 0.05),
        num_training_steps=total_steps
    )
    epochs = []
    epochs_loss = []
    epochs_accuracy = []
    for i in progress(options.epochs):
        timer = TrainingTimer(
            minutes=10
        )
        sum_loss = 0
        accuracy = 0
        rows = 0
        iterator = tqdm(loader)
        for X, y in iterator:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_predicted = model(X)

            loss = torch.nn.functional.cross_entropy(
                y_predicted.view(-1, config.vocab_size),
                y.view(-1),
                ignore_index=config.padding_index,
            )
            optimizer.zero_grad()
            loss.backward()
          #  torch.nn.utils.clip_grad_norm_(model.parameters(), options.max_grad_norm)
            optimizer.step()
         #   scheduler.step()
            sum_loss += loss.item()
            # Accuracy metrics
            y_sample_next = temperature_sampling(y_predicted[:, -1, :])
            y_next = y[:, -1]
            assert y_sample_next.shape == y_next.shape
            assert y_sample_next.shape == y_next.shape
            accuracy += (y_sample_next == y_next).sum()
            rows += y_next.shape.numel()
            if timer.done():
                break
            iterator.set_description("Accuracy {acc}, Loss {loss}".format(
                acc=(accuracy / rows * 100), 
                loss=sum_loss
            ))
        
        state_saver.save(model, optimizer, i, sum_loss)
        acc = accuracy / rows * 100
        epochs_accuracy.append(acc.item())
        epochs_loss.append(loss.item())
        epochs.append(i)
        assert acc <= 100, acc
        debug_print(f"epoch: {i}, loss: {sum_loss}, accuracy {acc}")

        with torch.no_grad():
            rows = dataset.sample(n=2)
            for i, j in rows:
                i, j = i.to(DEVICE), j.to(DEVICE)
                i = i.reshape((1, -1))
                predicted = model(i)[0]
                word_idx = temperature_sampling(predicted, temperature=0)

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
                y_sample_next = temperature_sampling(predicted[:, -1, :])
                accuracy += (y_sample_next.item() == y[-1].item())
            print((accuracy / index) * 100)

    return (epochs, epochs_accuracy, epochs_loss, model)


if __name__ == "__main__":
    tokenizer, cached = SimpleTextEncoder("example").load_cache()
    if not cached:
        print("Not cached building tokenizer.")
        tokenizer = tokenizer.build_from_files([
            "example.text"
        ])
        tokenizer.save_cache()
    else:
        print("Tokenizer is cached.")
    text_dataset = TransformerTextDataset.from_file(tokenizer, "example.text", sequence_length=4)
    train(
        text_dataset,
        options=TrainingOptions(
            batch_size=256
        ),
        progress=lambda x: tqdm(range(x))
    )
