import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_creator_list.dataloader import DocumentRankDataset
from torch.utils.data import DataLoader
from optimization_playground_shared.plot.Plot import Plot, Figure
from tqdm import tqdm
from typing import List
from results import Results, Input
from utils import rollout_model_binary
from best_model import BestModel
from dataset_creator_list.embedding_backend import EmbeddingBackend

class Model(nn.Module):
    def __init__(self, embeddings_size) -> None:
        super().__init__()
        self.base_layers = nn.Sequential(*[
            nn.Linear(embeddings_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
        ])
        n = 2
        self.output_layers = nn.Sequential(*[
            nn.Linear(256 * n, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ])
        self.model_file = ".ranknet.torch"

    def forward(self, item_1, item_2):
        delta = torch.concat((
            self.base_layers(item_1),
            self.base_layers(item_2),
        ), dim=1)
        return self.output_layers(delta)

    def label(self, item_1, item_2):
        # 1 if item_1 is the best, 0 if item_2 is the best
        return (self.forward(item_1, item_2) >= torch.tensor(0.5)).long()

    def rollout(self, items: List[Input]) -> List[Results]:
        return rollout_model_binary(self, items)
    
    def save(self):
        torch.save({
            "model_state": self.state_dict(),
        }, self.model_file)
    
    def load(self):
        state = torch.load(self.model_file, weights_only=True)
        self.load_state_dict(state["model_state"])
        return self

if __name__ == "__main__":
    batch_size = 32
    epochs = 1_000

    embedding_backend = EmbeddingBackend()
    model = Model(embeddings_size=embedding_backend.embedding_size())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_dataset = DocumentRankDataset(train=True, embedding_backend=embedding_backend.backend, dataset_format="binary")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DocumentRankDataset(train=False, embedding_backend=embedding_backend.backend, dataset_format="binary")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    best_model = BestModel()

    epoch_accuracy = []
    epoch_loss = []
    epoch_test_accuracy = []
    iterator = tqdm(range(epochs))
    for _ in iterator:
        sum_accuracy = torch.tensor(0.0)
        sum_loss = torch.tensor(0.0)
        count = torch.tensor(0.0)
        for (x, y, label) in train_loader:
            predicted = model(x, y)
            model.zero_grad()
            loss = nn.BCELoss()(predicted, label)   
            loss.backward()
            optimizer.step()
            
            pseudo_label = model.label(x, y)        
            assert pseudo_label.shape[0] == x.shape[0]
            assert label.shape[0] == x.shape[0]

            # accuracy = (pseudo_label == label).long().sum() / label.shape[0]  * 100
            sum_loss += loss
            sum_accuracy += (pseudo_label == label).long().sum()
            count += label.shape[0]
        epoch_loss.append(sum_loss.item())
        epoch_accuracy.append(sum_accuracy / count * 100)
        with torch.no_grad():
            sum_accuracy = torch.tensor(0.0)
            count = torch.tensor(0.0)
            for (x, y, label) in test_loader:
                pseudo_label = model.label(x, y)        
                assert pseudo_label.shape[0] == x.shape[0]
                assert label.shape[0] == x.shape[0]

                sum_accuracy += (pseudo_label == label).long().sum()
                count += label.shape[0]
            test_acc = sum_accuracy / count * 100
            epoch_test_accuracy.append(test_acc)
            best_model.set_model(model, test_acc)
            model = best_model.get_model()
        iterator.set_description(f"Training acc: {epoch_accuracy[-1]}, testing acc: {epoch_test_accuracy[-1]}, loss {epoch_loss[-1]}")

    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Loss": epoch_loss,
                },
                title="Training loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
            Figure(
                plots={
                    "Training accuracy": epoch_accuracy,
                },
                title="Accuracy",
                x_axes_text="Epochs",
                y_axes_text="accuracy",
            ),
            Figure(
                plots={
                    "Testing accuracy": epoch_test_accuracy,
                },
                title="(test) Accuracy",
                x_axes_text="Epochs",
                y_axes_text="accuracy",
            ),
        ],
        name=f'training_ranknet.png'
    )
    print(f"Finale model score: {best_model.score}%")
    model.save()
