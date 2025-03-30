from model import Model, Config
from transformer_dataset import TransformerDataset, TransformerTextDataset
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
)
from typing import Callable


def create_config(vocab_size, padding_index, sequence_length):
    return Config(
        sequence_length=sequence_length,
        dim_embeddings=32,
        num_attention_heads=4,
        num_transformer_layers=1,
        padding_index=padding_index,
        vocab_size=vocab_size,
    )

def train(
    dataset: TransformerDataset, 
    override: Callable[[Config], Config] = (lambda x: x),
    create_model: Callable[[Config], Model] = (lambda x: Model(x))
):
    config = create_config(
        dataset.vocab_size,
        dataset.padding_index,
        sequence_length=dataset.sequence_size,
    )
    config = override(config)
    model = create_model(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = dataset.iter(
        batch_size=32,
    )
    epochs = []
    epochs_loss = []
    epochs_accuracy = []
    for i in range(1_000):
        sum_loss = 0
        accuracy = 0
        rows = 0
        for X, y in loader:
            y_predicted = model(X)
            loss = torch.nn.functional.cross_entropy(
                y_predicted.view(-1, config.vocab_size),
                y.view(-1),
                ignore_index=config.padding_index,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            y_sample_next = temperature_sampling(y_predicted[:, -1, :])
            y_next = y[:, -1]
            assert y_sample_next.shape == y_next.shape
            assert y_sample_next.shape == y_next.shape
            accuracy += (y_sample_next == y_next).sum()
            rows += y_next.shape.numel()
        acc = accuracy / rows * 100
        epochs_accuracy.append(acc.item())
        epochs_loss.append(loss.item())
        epochs.append(i)
        assert acc <= 100, acc
        print(f"epoch: {i}, loss: {sum_loss}, accuracy {acc}")

        rows = dataset.sample(n=2)
        for i, j in rows:
            i = i.reshape((1, -1))
            predicted = model(i)[0]
            word_idx = temperature_sampling(predicted)

            next_word_idx = word_idx[-1]
            expected_word_idx = j[-1]
            context = "".join([dataset.decode(idx) for idx in i[0]])
            print(f"\tcontext: {context}")
            word = dataset.decode(next_word_idx)
            expected = dataset.decode(expected_word_idx.item())
            print(f"\tnext token: '{word}'")
            print(f"\texpected token: '{expected}'")
            print("")
    return (epochs, epochs_accuracy, epochs_loss)


if __name__ == "__main__":
    text_dataset = TransformerTextDataset.from_file("example.text", sequence_length=4)
    train(text_dataset)
