from reference.minigpt_model import GPT, GPTConfig
from utils.web_dataloader import WebDataloader
import os
import torch
import torch.optim as optim
from dotenv import load_dotenv
from training.objectives import NextTokenPrediction
from optimization_playground_shared.nlp.utils.sampling import (
    argmax_sampling,
)

load_dotenv()


def train():
    dataset = WebDataloader(os.environ["WEB_DATALOADER"], "small-web", batch_size=128)
    device = torch.device("cuda:1")
    model = GPT(
        GPTConfig(
            vocab_size=dataset.vocab_size,
            block_size=dataset.sequence_size,
            n_layer=4,
            n_embd=128,
            n_head=4,
            bias=False,
        )
    ).to(device)
    evaluator = NextTokenPrediction(
        padding_index=dataset.padding_index,
        vocab_size=dataset.vocab_size,
        sampler=argmax_sampling,
    )
    optimizer = optim.Adam(model.parameters())
    dataloader = dataset.iter()

    loss_total = 0
    accuracy_total = 0
    accuracy_rows = 0
    for i in range(100):
        print(f"Epoch {i}")
        for index, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred, loss = model(X, y)
            loss.backward()
            optimizer.step()

            acc, _rows = evaluator.evaluator(y_pred, y)
            accuracy_total += acc
            accuracy_rows += _rows
            loss_total += loss
            if index % 100 == 0 and index > 1:
                print((loss_total / index), (accuracy_total / accuracy_rows) * 100)


if __name__ == "__main__":
    train()
