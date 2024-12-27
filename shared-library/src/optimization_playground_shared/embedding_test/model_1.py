import torch
import torch.nn as nn
import torch.optim as optim
from ..apis.url_to_text import get_url_documents
from ..nlp.DocumentEncoderSequence import SimpleVocab, get_document_dataset
import os

import torch
import torch.nn.functional as F

class SimpleContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, epsilon=1e-6):
        super(SimpleContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        positives = torch.diagonal(similarity_matrix)

        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        negatives = similarity_matrix.masked_fill(mask, float('-inf'))
        negatives = torch.logsumexp(negatives, dim=1)

        # Todo: figure out why things turn negative
        loss = (negatives -positives).abs()

        return loss.mean()

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(*[
             nn.Linear(embed_dim, 1024),
             nn.ReLU(),
             nn.Linear(1024, 2048),
             nn.ReLU(),
             nn.Linear(2048, 1024),
             nn.ReLU(),
             nn.Linear(1024, 512),
#             nn.ReLU(),
        ])

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)
        return self.fc(embedded)

class EmbeddingModelOne:
    def __init__(self):
        self.document_encoder = SimpleVocab()
        self.model = None

    def loss_contrastive(self, logits):
        # return SimpleContrastiveLoss().forward(logits)
        labels = torch.arange(logits.shape[0])
        l_row = torch.nn.CrossEntropyLoss()(logits, labels)
        return l_row

    def train(self, docs):
        documents_encoder = self.document_encoder.fit(docs)
        self.document_encoder.lock()

        embed_dim = 64
        lr = 0.001
        num_epochs = 512
        batch_size = 512

        self.model = SimpleEmbeddingModel(documents_encoder.size, embed_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        inputs = get_document_dataset(documents_encoder, docs, SEQUENCE_LENGTH=512)

        for epoch in range(num_epochs):
            # batch size = 32
            for i in range(0, inputs.shape[0], batch_size):
                self.model.train()
                outputs = self.model(inputs[i:i+batch_size])
                loss = self.loss_contrastive(outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        print("Training completed!")
        return self

    def get_embedding(self, docs):
        embedings = []
        for text in docs:
            inputs = get_document_dataset(self.document_encoder, [text], SEQUENCE_LENGTH=512)
            embedding = torch.zeros((512))
            batch_size = 512
            self.model.eval()
            with torch.no_grad():
                for i in range(0, inputs.shape[0], batch_size):
                    output = self.model(inputs[i:i+batch_size]).mean(dim=0)
                    embedding += output
            embedings.append(embedding / inputs.shape[0])
        return embedings

    def save(self):
        path = os.path.join(os.path.dirname(
            __file__
        ), ".model")
        os.makedirs(path, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "vocab_size": self.model.vocab_size,
            "embed_dim": self.model.embed_dim,
        }, os.path.join(
            path,
            "model.pth"
        ))
        self.document_encoder.save(path)
    
    def load(self):
        path = os.path.join(os.path.dirname(
            __file__
        ), ".model")
        os.makedirs(path, exist_ok=True)
        model_data = torch.load(
            os.path.join(
                path,
                "model.pth"
            )
        )
        self.model = SimpleEmbeddingModel(
            model_data["vocab_size"],
            model_data["embed_dim"]
        )
        self.model.load_state_dict(model_data["state_dict"])
        self.model.eval()
        self.document_encoder = self.document_encoder.load(path)

if __name__ == "__main__":
    model = EmbeddingModelOne().train(
        get_url_documents()
    )
    model.save()
    model.load()
    print(model.encode_document("hello, this is some text"))
