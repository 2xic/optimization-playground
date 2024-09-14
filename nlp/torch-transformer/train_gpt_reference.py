"""
Using https://github.com/karpathy/nanoGPT as a reference

Mostly using info form https://github.com/karpathy/nanoGPT/blob/master/train.py

"""
import math

from optimization_playground_shared.nlp.utils.sampling import temperature_sampling
from reference.minigpt_model import GPT, GPTConfig
from train_gpt import source_vocab, get_dataloader
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
import torch.optim as optim
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.metrics_tracker.metrics import Prediction
import torch
import random
import json
from tqdm import tqdm

learning_rate = 6e-4 # max learning rate

X, y = get_dataloader()

model = GPT(
    GPTConfig(
        block_size=64,
        vocab_size=source_vocab.size,
        n_layer=4,
        n_head=4,
        n_embd=256,
    )
)

dataloader = get_raw_dataloader((
    X.clone(),
    y.clone()
),
    batch_size=32,
    shuffle=False,
)

def rollout(model: GPT, seed, steps):
    sequence_size = 64
    with torch.no_grad():
        output = []
        prev = None
        for index in range(steps):
            next_predicted = None
            if len(seed) <= index:
                X = torch.full((1, sequence_size), -1).reshape(1, -1).to('cuda').long()
                context_tensor = torch.tensor(output[-sequence_size:]).long()
                X[0, :context_tensor.shape[0]] = context_tensor
                assert prev is None or torch.all(prev[0, 1:] == X[0, :-1])

                next_token, _ = model.forward(X)
                next_predicted = temperature_sampling(
                    next_token.reshape((next_token.shape[0], -1))[0]
                ).item()
                assert type(next_predicted) == int, next_predicted
                output.append(next_predicted)
                prev = X
            else:
                next_predicted = seed[index].item()
                assert type(next_predicted) == int, next_predicted
                output.append(next_predicted)
        return output

def get_text_prediction(model, seed: torch.Tensor):
    results = []
    raw_tokens = []
    with torch.no_grad():
        y = rollout(
            model,
            seed=seed,
            steps=512,
        )
        for index, i in enumerate(y):
            if index == seed.shape[-1]:
                results.append("*model output start*")
                raw_tokens.append(-42)
            results.append(source_vocab.vocab.index_vocab[i])
            raw_tokens.append(i)
    return " ".join(results), raw_tokens


epochs = 1024
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

metrics_tracker = Tracker("train_gpt_shakespeare_reference") if __name__ == "__main__" else None

model.to('cuda')

for epoch in range(epochs):
    sum_loss = 0
    for (batch_X, batch_y) in tqdm(dataloader):
        batch_X = batch_X.to('cuda')
        batch_y = batch_y.to('cuda')
        optimizer.zero_grad()
        logits, loss = model(batch_X, batch_y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    text, raw_tokens = get_text_prediction(model, X[random.randint(0, X.shape[0] - 1)])
    metrics_tracker.log(
        Metrics(
            epoch=epoch,
            loss=sum_loss,
            training_accuracy=None,
            prediction=Prediction.text_prediction(
                "\n".join([
                    "text: ",
                    text,
                    "tokens: ",
                    json.dumps(raw_tokens)
                ])
            )
        )
    )