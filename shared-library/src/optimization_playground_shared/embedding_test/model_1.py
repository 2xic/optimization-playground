import torch
import torch.nn as nn
import torch.optim as optim
from ..apis.url_to_text import get_url_documents
from ..nlp.DocumentEncoderSequence import get_document_dataset
import torch.nn.functional as F
from .base_model import BaseModel

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

class NegativeSample(torch.nn.Module):
    def __init__(self, temperature=0.1, epsilon=1e-6):
        super(NegativeSample, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, doc_embeddings):
        num_docs = doc_embeddings.shape[0]
        dot_product_matrix = torch.matmul(doc_embeddings, doc_embeddings.T)
        positive_scores = torch.diagonal(dot_product_matrix)
        negative_scores = dot_product_matrix - torch.eye(num_docs, device=doc_embeddings.device) * dot_product_matrix
        positive_loss = F.logsigmoid(positive_scores)
        negative_loss = F.logsigmoid(-negative_scores)
        loss = - (positive_loss + negative_loss.sum(dim=1)).mean()
    
        return loss

class MinimalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, epsilon=1e-6):
        super(MinimalCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, logits):
        labels = torch.arange(logits.shape[0])
        l_row = torch.nn.CrossEntropyLoss()(logits, labels)
        return l_row

    
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
        ])

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)
        return self.fc(embedded)
    


class EmbeddingModelOne(BaseModel):
    def __init__(self, loss=""):
        self.loss_functions = {
            "MinimalCrossEntropyLoss": MinimalCrossEntropyLoss(),
            "SimpleContrastiveLoss": SimpleContrastiveLoss(),
            "NegativeSample": NegativeSample(),
        }
        self.loss = self.loss_functions[loss]
        super().__init__()

    def train(self, docs):
        documents_encoder = self.document_encoder.fit(docs)
        self.document_encoder.lock()

        lr = 0.001
        embed_dim = 64
        num_epochs = 512
        batch_size = 512

        self.model = SimpleEmbeddingModel(documents_encoder.size, embed_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        inputs = get_document_dataset(documents_encoder, docs, SEQUENCE_LENGTH=512)

        for epoch in range(num_epochs):
            for i in range(0, inputs.shape[0], batch_size):
                self.model.train()
                outputs = self.model(inputs[i:i+batch_size])
                loss = self.loss(outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        print("Training completed!")
        return self

    def _load_model(self, model_data):
        return SimpleEmbeddingModel(
            model_data["vocab_size"],
            model_data["embed_dim"]
        )
    
    def _get_state_dict(self):
         return {
            "state_dict": self.model.state_dict(),
            "vocab_size": self.model.vocab_size,
            "embed_dim": self.model.embed_dim,
        }

if __name__ == "__main__":
    model = EmbeddingModelOne().train(
        get_url_documents()
    )
    model.save()
    model.load()
    print(model.transforms("hello, this is some text"))
