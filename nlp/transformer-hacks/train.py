from model import Model, Config
from transformer_dataset import TextDataset
import os 
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling, argmax_sampling

def train():
    config = Config(
        sequence_length=8,
        dim_embeddings=8,
        vocab_size=-1,
        num_transformer_layers=4
    )
    dataset = TextDataset.from_file("example.text", config)
    #.from_folder(
     #   os.path.dirname(__file__) + "/*",
     #   config
    #)
    config.vocab_size = dataset.vocab_size
    model = Model(config)
    optimizer = optim.Adam(model.parameters())
    loader = dataset.iter(
        batch_size=32
    )

    for i in range(1_000):
        sum_loss = 0
        accuracy = 0
        rows = 0
        for (X, y) in loader:
            y_predicted = model(X)
            loss = torch.nn.functional.cross_entropy(
                y_predicted.view(-1, config.vocab_size),
                y.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            rows += X.shape[0]
            y_sample = argmax_sampling(
                y_predicted.view(-1, config.vocab_size)
            )
            accuracy += (y_sample[::config.sequence_length] == y.view(-1)[::config.sequence_length]).sum()
        print(f"index: {i}, loss: {sum_loss}, accuracy {accuracy / rows * 100}")
    
        rows = dataset.sample(n=2)
        for (i, j) in rows:
            i = i.reshape((1, -1))
            predicted = model(i)[-1]
            word_idx = temperature_sampling(predicted[0])
            context = ([
                dataset.encoder.decode_idx(idx)
                for idx in i[0]
            ])
            print("".join(context))
            word = dataset.encoder.decode_idx(word_idx)
            expected = dataset.encoder.decode_idx(j[-1].item())
            print(f"next token: '{word}'")
            print(f"expected token: '{expected}'")
            print("")

if __name__ == "__main__":
    train()
