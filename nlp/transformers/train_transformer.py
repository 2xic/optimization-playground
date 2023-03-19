import torch.nn as nn
import torch
from utils import Tokenizer

def train_transformer(Transformer):
    documents = [
        "test sentence hello sir",
        "this is also a sentence",
        "what is a hello",
        "what is a sentence"
    ]
    targets = [
        1,
        1,
        0,
        0,
    ]
    tokenizer = Tokenizer().encode_documents(documents)
    model = Transformer(
        tokens=len(tokenizer.word_idx)
    )
    optimizer = torch.optim.Adam(
        model.parameters()
    )
    X = torch.concat(
        [tokenizer.encode_document_tensor(x, sequence_length=4) for x in documents]
    )
    y_truth_labels = torch.concat(
        [tokenizer.encode_document_tensor(x, sequence_length=4).flip(dims=(1, ))
        for x in documents]
    )
    y_truth = y_truth_labels.view(-1)
    y_pred = model(X)

    losses = []
    accuracy = []

    for _ in range(10_000):
        y_pred = model(X)
        prediction = y_pred.view(-1,y_pred.size(-1))
        
        optimizer.zero_grad()
        error = nn.CrossEntropyLoss()(
            prediction.float(),
            y_truth
        )
        error.backward()
        optimizer.step()
        
        acc = ((y_pred.argmax(dim=-1) == y_truth_labels).float().sum(dim=1)).mean()
        accuracy.append(acc)
        losses.append(error.item())

    model.eval()
    print(
        documents
    )        
    print(
        [
            tokenizer.decode(i) for i in model(X).argmax(dim=-1)
        ]
    )
    return losses, accuracy
