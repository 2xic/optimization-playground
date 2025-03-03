import torch.nn as nn
import torch
from utils import Tokenizer
from tqdm import tqdm
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling

def train_transformer(Transformer):
    documents = [
        "test sentence hello sir",
        "this is also a sentence",
        "what is a hello",
        "what is a sentence"
    ]
    tokenizer = Tokenizer().encode_documents(documents)
    model = Transformer(
        num_tokens=len(tokenizer.word_idx),
        sequence_size=4,
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

    for _ in tqdm(range(1000)):
        y_pred = model(X)
        prediction = y_pred.view(-1, len(tokenizer.idx_words))
        
        optimizer.zero_grad()
        error = nn.functional.cross_entropy(
            prediction.float(),
            y_truth
        )
        error.backward()
        optimizer.step()
        
        acc = ((y_pred.argmax(dim=-1) == y_truth_labels).float().sum(dim=1)).mean()
        accuracy.append(acc)
        losses.append(error.item())

    model.eval()
    print(accuracy[-5:])
    print(losses[-5:])

    print("Expected vs generated")
    print(
        [
            tokenizer.decode(i) for i in y_truth_labels
        ]
    )    
    out = model(X).view(-1, len(tokenizer.idx_words))
    print(
        tokenizer.decode(torch.tensor([
            temperature_sampling(v)
            for v in out[:4]
        ]))
    )
    return losses, accuracy
