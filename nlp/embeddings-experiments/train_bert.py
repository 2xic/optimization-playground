from shared.get_sentence_sim import get_sentence_sim
from shared.process import Process
from shared.vocab import Vocab
from shared.duplicate_tensor_batch import duplicate_tensor_batch
from bert.model import BertModel
from bert.Dataloader import TransformerDataset
from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    raw_text_dataset = [
        "taco is a food",
        "This a long sentence about how I love taco",
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

    dataset = TransformerDataset(
        vocab, dataset
    )
    batch_size = 32
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

    model = BertModel(vocab, device,nlayers=2)
    optimizer = torch.optim.Adam(
        model.parameters()
    )

    print(f"dataset == {len(dataset)}")

    for epoch in range(10_000):
        for index, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.fit(
                duplicate_tensor_batch(X.to(device), batch_size) , 
                duplicate_tensor_batch(y.to(device), batch_size)
            )
            loss.backward()
            optimizer.step()
            if index % 1_00 == 0:
                print(epoch, loss)
#    print(X.shape)
    model.fit(X.to(device), y.to(device), debug=True)
