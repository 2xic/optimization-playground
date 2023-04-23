from shared.process import Process
from shared.vocab import Vocab
from Cbow.Dataloader import CbowDataset
from Cbow.Model import CbowModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def get_sentence_sim(a, b):
    sentence_a = (model.predict(a))
    sentence_b = (model.predict(b))
    #print(sentence_a)
    #print(sentence_b)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return (f"{a} / {b} > sim: {cos(sentence_a, sentence_b).item()}")

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#    device = torch.device('cpu')
    raw_text_dataset = [
        "This a long sentence about how I love tacos",
        "taco is a food",
        "people like having tacos for dinner",
        "Tacos is nice, but what about the evm ? ",
        "The evm is a great machine, and I like it",
        "symbolic execution on the evm is cool",
        "the evm is build for ethereum , bitcoin does not use it",
        "bitcoin and ethereum are two great protocols",
        "I once saw someone use bitcoin and ethereum , they smiled",
        "I dislike how people forget bitcoin have smart contracts",
        "ethereum and bitcoin both have smart contracts",
    ]
    text_dataset = list(map(lambda x: Process().process(x), raw_text_dataset))

    vocab = Vocab()
    dataset = []
    for i in text_dataset:
        vocab.fit(i)

    for i in text_dataset:
        dataset.append(vocab.add(i).get(i))
    print(len(text_dataset))
    print(len(dataset))

    dataset = CbowDataset(vocab, dataset)
    print(len(dataset))
    dataloader = DataLoader(dataset, 
                            batch_size=128,
                            shuffle=True, 
                            num_workers=4)
    model = CbowModel(vocab, device)
    print(f"dataset == {len(dataset)}")

    for epoch in range(1_000):
        for index, (X, y) in enumerate(dataloader):
            loss = model.fit(X.to(device), y.to(device))
            if index % 1_00 == 0:
                print(epoch, loss)

    print(
        get_sentence_sim(
            "taco",
            "ethereum"
        )
    )
    print(
        get_sentence_sim(
            "bitcoin",
            "ethereum"
        )
    )
    print(
        get_sentence_sim(
            "bitcoin",
            "taco"
        )
    )
