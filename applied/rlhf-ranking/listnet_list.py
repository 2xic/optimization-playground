"""
https://embracingtherandom.com/machine-learning/tensorflow/ranking/deep-learning/learning-to-rank-part-2/

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_creator_list.dataloader import DocumentRankDataset
from torch.utils.data import DataLoader
from optimization_playground_shared.plot.Plot import Plot, Figure
from tqdm import tqdm
from results import Input, Results
from typing import List
from best_model import BestModel

class Model(nn.Module):
    def __init__(self, embeddings_size) -> None:
        super().__init__()

        self.base_layers = nn.Sequential(*[
            nn.Linear(embeddings_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        ])
        n = 5
        self.output_layers = nn.Sequential(
            nn.Linear(256 * n, 128),
            nn.Sigmoid(),
            nn.Linear(128, n),
            nn.Softmax(dim=1),
        )
        self.model_file = ".listnet_list.torch"

    def forward(self, item_1, item_2, item_3, item_4, item_5):
        delta = torch.concat((
                self.base_layers(item_1),
                self.base_layers(item_2),
                self.base_layers(item_3),
                self.base_layers(item_4),
                self.base_layers(item_5)
        ), dim=1)
        return self.output_layers(delta)

    def label(self, item_1, item_2, item_3, item_4, item_5):
        return (self.forward(item_1, item_2, item_3, item_4, item_5) >= torch.tensor(0.5)).long()
    
    def rollout(self, items: List[Input]) -> List[Results]:
        with torch.no_grad():
            scores: List[Results] = []
            results = self.forward(*list(map(lambda x: x.item_tensor, items)))[0]
            for index, i in enumerate(items):
                scores.append(Results(
                    item_id=i.item_id,
                    item_score=results[index].item(),
                    item_tensor=i.item_tensor,
                ))
            return scores

    def save(self):
        torch.save({
            "model_state": self.state_dict(),
        }, self.model_file)
    
    def load(self):
        state = torch.load(self.model_file)
        self.load_state_dict(state["model_state"])
        return self
    
if __name__ == "__main__":
    batch_size = 256
    model = Model(embeddings_size=1536)
    optimizer = torch.optim.Adam(model.parameters())
    train_dataset = DocumentRankDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DocumentRankDataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    best_model = BestModel()

    epoch_accuracy = []
    epoch_loss = []
    epoch_test_accuracy = []
    iterator = tqdm(range(1_00))
    for _ in iterator:
        sum_accuracy = torch.tensor(0.0)
        sum_loss = torch.tensor(0.0)
        count = torch.tensor(0.0)
        for (item_1, item_2, item_3, item_4, item_5, label) in train_loader:
            predicted = model(item_1, item_2, item_3, item_4, item_5)
            model.zero_grad()

            loss = nn.functional.kl_div(predicted.log(), label, reduction="batchmean")
            loss.backward()
            optimizer.step()

            # assert not torch.all(torch.argmax(label, dim=1) == 0)            
            pseudo_label = torch.argmax(model.label(item_1, item_2, item_3, item_4, item_5), dim=1)
            label = torch.argmax(label, dim=1)

            sum_loss += loss
            sum_accuracy += (pseudo_label == label).long().sum()
            count += label.shape[0]

        epoch_loss.append(sum_loss.item())
        epoch_accuracy.append(sum_accuracy / count * 100)

        with torch.no_grad():
            sum_accuracy = torch.tensor(0.0)
            count = torch.tensor(0.0)
            for (item_1, item_2, item_3, item_4, item_5, label) in test_loader:
                pseudo_label = torch.argmax(model.label(item_1, item_2, item_3, item_4, item_5), dim=1)
                label = torch.argmax(label, dim=1)

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
        name=f'training_listnet_list.png'
    )
    print(f"Finale model score: {best_model.score}%")
    model.save()
