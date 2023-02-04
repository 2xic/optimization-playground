from components.model import Net
from dataloader import Cifar10Dataloader
import torch
from torch.utils.data import DataLoader
from eval import eval_model
import torch.nn.functional as F
from components.parameters import EPOCHS, BATCH_SIZE
from components.predictor import Predictor
from components.combined import CombinedModel

def train(lock):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    student = Net().to(device)
    predictor = Predictor()

    model = CombinedModel([
            student,
            predictor
    ]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4
    )

    dataloader = DataLoader(
        Cifar10Dataloader(),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    with open("logs/train_model_without_features.txt", "w") as file:
        for epoch in range(EPOCHS):
            total_loss = torch.tensor(0.0, device=device)
            for (X, y) in dataloader:
                X = X.to(device)
                y = y.to(device)

                value = torch.nn.NLLLoss()(
                    F.log_softmax(model(X), dim=1),
                    y
                )

                optimizer.zero_grad()
                value.backward()
                optimizer.step()        

                total_loss += value.item()

            acc = eval_model(model)
            lock.acquire()
            try:
                print(f"no features - epoch: {epoch}, total_loss: {total_loss}, acc: {acc}")
            finally:
                lock.release()
            file.write(f"{acc}\n")

if __name__ == "__main__":
    from torch.multiprocessing import Lock
    lock = Lock()
    train(lock)
