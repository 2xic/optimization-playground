import torch
import torch.nn as nn
import torch.optim as optim
from ..apis.url_to_text import get_url_documents
from sklearn.feature_extraction.text import CountVectorizer
import random

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)

class DocumentEncoder:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = -1

    def fit(self, documents):
        tokens = sum([
            self.get_tokens(docs)
            for docs in documents
        ], [])
        self.vocab = {char: idx for idx, char in enumerate(list(set(tokens)))}
        self.vocab_size = len(self.vocab) + 1
        return self
    
    def transformer(self, text):
        tokens =  self.get_tokens(text)
        batch = 512
        offset = random.randint(0, len(tokens) - batch)
        return torch.tensor([
            self.vocab.get(token, len(self.vocab)) for token in tokens[offset:offset+batch]
        ], dtype=torch.long)

    def get_tokens(self, document):
        return document.split(" ")

def get_data_from_documents(documents: DocumentEncoder, docs):
    sequences = [documents.transformer(doc) for doc in docs]
    padded_data = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
  #  print(padded_data.shape)
    inputs = padded_data[:, :-1]
    targets = padded_data[:, 1:]
    return inputs, targets

def train_model(documents_encoder: DocumentEncoder, docs):
    documents_encoder = documents_encoder.fit(docs)

    print(len(docs))
    embed_dim = 64
    hidden_dim = 128
    lr = 0.001
    num_epochs = 32

    model = SimpleEmbeddingModel(documents_encoder.vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        step = 2
        for i in range(0, len(docs)- step, step):
            model.train()
            inputs, targets = get_data_from_documents(documents_encoder, docs[i:(i+step)])
            outputs = model(inputs)
            loss = criterion(outputs.reshape((-1, documents_encoder.vocab_size)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Training completed!")

if __name__ == "__main__":
    document_encoder = DocumentEncoder()
    train_model(
        document_encoder,
        get_url_documents()
    )
