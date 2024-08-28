import torch
from dataset_creator_list.dataloader import DocumentRankDataset
from torch.utils.data import DataLoader
from semi_supervised_machine import SemiSupervisedMachine
from tqdm import tqdm

if __name__ == "__main__":
    batch_size = 8
    model = SemiSupervisedMachine()
    test_dataset = DocumentRankDataset(train=False, dataset_format="softmax")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    sum_accuracy = torch.tensor(0.0)
    count = torch.tensor(0.0)
    first_item_accuracy = torch.tensor(0.0)
    first_item_count = torch.tensor(0.0)
    with torch.no_grad():
        for (item_1, item_2, item_3, item_4, item_5, label) in tqdm(test_loader):
            X = torch.zeros((
                item_1.shape[0],
                5,
                item_1.shape[1],
            ))
            X[:, 0, :] = item_1[:]
            X[:, 1, :] = item_2[:]
            X[:, 2, :] = item_3[:]
            X[:, 3, :] = item_4[:]
            X[:, 4, :] = item_5[:]
            predicted_label = model.rank_documents_tensor(X)
            order1 = torch.argsort(predicted_label, descending=True, dim=0)
            order2 = torch.argsort(label, descending=True, dim=0)
            sum_accuracy += (order1 == order2).long().sum()
            count += order1.shape[0] * order1.shape[1]

            order1 = torch.argmax(predicted_label, dim=0)
            order2 = torch.argmax(label, dim=0)
            first_item_accuracy += (order1 == order2).long().sum()
            first_item_count += order1.shape[0]

    print(f"Accuracy {sum_accuracy / count}")
    print(f"First item accuracy {first_item_accuracy / first_item_count}")

