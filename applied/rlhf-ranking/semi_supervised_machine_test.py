import torch
from dataset_creator_list.dataloader import DocumentRankDataset
from torch.utils.data import DataLoader
from semi_supervised_machine import SemiSupervisedMachine
from tqdm import tqdm
from collections import defaultdict

def verify_accuracy():
    batch_size = 1
    model = SemiSupervisedMachine()
    model.models = model.models[:1]
    test_dataset = DocumentRankDataset(train=False, dataset_format="softmax", row_size=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        for items in tqdm(test_loader):
            X = torch.zeros((
                items[0].shape[0],
                test_dataset.row_size,
                items[0].shape[1],
            ))
            predicted_label = model.rank_documents_tensor(X)   
            # Swapped for how we rank with the semi supervised version.
            reference_model_predicted_label = model.models[0](
                X[:, 1, :],
                X[:, 0, :],
            )
            if reference_model_predicted_label[0] < 0.5:
                reference_model_predicted_label = torch.tensor([[0, 1]])
            else:
                reference_model_predicted_label = torch.tensor([[1, 0]])
            
            assert torch.all(torch.argsort(
                predicted_label,
            ) == torch.argsort(
                reference_model_predicted_label,
            ))


if __name__ == "__main__":
    verify_accuracy()
    batch_size = 32
    model = SemiSupervisedMachine()
    test_dataset = DocumentRankDataset(train=False, dataset_format="softmax")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    sum_accuracy = torch.tensor(0.0)
    count = torch.tensor(0.0)
    first_item_accuracy = torch.tensor(0.0)
    first_item_count = torch.tensor(0.0)
    n_item_accuracy = defaultdict(list)
    with torch.no_grad():
        for items in tqdm(test_loader):
            X = torch.zeros((
                items[0].shape[0],
                test_dataset.row_size,
                items[0].shape[1],
            ))
            for i in range(test_dataset.row_size):
                X[:, i, :] = items[i][:]
            label = items[-1]
            predicted_label = model.rank_documents_tensor(X)   

            order1 = torch.argsort(predicted_label, descending=True, dim=1)
            order2 = torch.argsort(label, descending=True, dim=1)
            sum_accuracy += (order1 == order2).long().sum()
            count += order1.shape[0] * order1.shape[1]

            for i in range(order1.shape[0]):
                for index in range(order1.shape[1]):
                    item_i = order1[i][index]
                    item_j = order2[i][index]
                    n_item_accuracy[index].append(item_i.item() == item_j.item())

    print(count)
    print(sum_accuracy)
    print(f"Accuracy {sum_accuracy / count}")
    for index, _ in enumerate(n_item_accuracy):
        v = n_item_accuracy[index]
        print(f"index {index} item accuracy {sum(v) / len(v) * 100}")
