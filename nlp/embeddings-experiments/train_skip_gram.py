from shared.get_sentence_sim import get_sentence_sim
from shared.process import Process
from shared.vocab import Vocab
from SkipGram.Dataloader import SkipGramDataset
from SkipGram.Model import SkipGramModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cache_file = "dataset.json"

    raw_text_dataset = [
        "This a long sentence about how I love taco",
        "taco is a food",
        "people like having taco for dinner",
        "Taco is nice, but what about the evm ? ",
        "The evm is a great machine, and I like it",
        "symbolic execution on the evm is cool",
        "the evm is build for ethereum , bitcoin does not use it",
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

    dataset = SkipGramDataset(vocab, dataset)
    print(len(dataset))
    dataloader = DataLoader(dataset, 
                            batch_size=128,
                            shuffle=True, 
                            num_workers=4)
    model = SkipGramModel(vocab, device)
    print(f"dataset == {len(dataset)}")

    for epoch in range(1_000):
        for index, (X, y) in enumerate(dataloader):
            loss = model.fit(X.to(device), y.to(device))
            if index % 1_00 == 0:
                print(epoch, loss)

    print(
        get_sentence_sim(
            model,
            "taco",
            "ethereum"
        )
    )
    print(
        get_sentence_sim(
            model,
            "bitcoin",
            "ethereum"
        )
    )
    print(
        get_sentence_sim(
            model,
            "bitcoin",
            "taco"
        )
    )
