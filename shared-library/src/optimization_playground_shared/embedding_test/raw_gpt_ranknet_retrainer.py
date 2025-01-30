"""
python3 -m optimization_playground_shared.embedding_test.raw_gpt_ranknet_retrainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .raw_gpt_runner_fast import ModelEmbeddings
from torch.utils.data import Dataset
import json
import random
import os 
from ..utils.RunHostedModel import ModelHost

class DocumentRankDataset(Dataset):
    def __init__(self):
        rows = []
        path = os.path.join(
            os.path.dirname(__file__),
            "dataset_ranked_documents.json"
        )
        with open(path, "r") as file:
            for i in json.load(file):
                for entries in range(0, len(i), 2):
                    a = i[entries:entries + 2]
                    if len(a) != 2:
                        continue
                    rows.append(a)
        self.rows = rows

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if random.randint(0, 1) == 0:
            return self.rows[idx][0], self.rows[idx][1], torch.tensor([1])
        else:
            return self.rows[idx][1], self.rows[idx][0], torch.tensor([0])

class Model(nn.Module):
    def __init__(self, backend) -> None:
        super().__init__()
        self.base_layers = nn.Sequential(*[
            nn.Linear(backend.model.embedding_size, 2048),
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

    def forward(self, item_1, item_2):
        delta = torch.concat((
            self.base_layers(item_1),
            self.base_layers(item_2),
        ), dim=1)
        return self.output_layers(delta)

    def label(self, item_1, item_2):
        # 1 if item_1 is the best, 0 if item_2 is the best
        return (self.forward(item_1, item_2) >= torch.tensor(0.5)).long()
    
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

    backend = ModelEmbeddings(
        "latests_gpt_raw_fast.pth"
    )
    backend.model.disable_require_grad_first_layer()
    backend.model.train()
    optimizer_embeddings = torch.optim.Adam(backend.model.parameters(), lr=1e-4)


    model = Model(backend)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader = DocumentRankDataset()

    epoch_accuracy = []
    epoch_loss = []
    epoch_test_accuracy = []
    iterator = tqdm(range(100))
    for _ in iterator:
        sum_accuracy = torch.tensor(0.0)
        sum_loss = torch.tensor(0.0)
        count = torch.tensor(0.0)
        for (x, y, label) in train_loader:
            predicted = model(
                backend.transforms([x]), 
                backend.transforms([y])
            )
            model.zero_grad()
            backend.model.zero_grad()
            loss = nn.BCELoss()(predicted, label.reshape(predicted.shape).float())   
            loss.backward()

            optimizer_embeddings.step()
            optimizer.step()
            
    # Load out the model
    host = ModelHost()
    host.add_model("model", backend)
    host.run()    

# python3 -m optimization_playground_shared.embedding_test.raw_gpt_ranknet_retrainer

