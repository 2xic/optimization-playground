from components.model import Net
from dataloader import Cifar10Dataloader
import torch
from torch.utils.data import DataLoader
from components.predictor import Predictor
from eval import eval_model
import torch.nn.functional as F
from components.parameters import EPOCHS, BATCH_SIZE, set_no_grad
from components.combined import CombinedModel

def train_byol_transfer_features():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    student = Net().to(device)
    predictor = Predictor()
    student.eval()
    student.load_state_dict(torch.load("student_model"))
    set_no_grad(student)

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

    with open("logs/train_model_with_byol_features_test_acc.txt", "w") as test_acc:
        with open("logs/train_model_with_byol_features_train_acc.txt", "w") as train_acc:
            for epoch in range(EPOCHS):
                total_loss = torch.tensor(0.0, device=device)
                training_acc = torch.tensor(0.0, device=device)
                rows = 0
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
                    training_acc += torch.sum(
                        y == torch.argmax(model(X), 1)
                    )
                    rows += X.shape[0]
                training_acc =  (training_acc / rows) * 100 
                acc = eval_model(model)
                print(f"BYOL features model - epoch: {epoch}, total_loss: {total_loss}, acc: {acc}")
                test_acc.write(f"{acc}\n")
                train_acc.write(f"{training_acc}\n")

if __name__ == "__main__":
    train_byol_transfer_features()
