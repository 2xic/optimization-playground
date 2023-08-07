from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class _RawTensorDataset(Dataset):
    def __init__(self, tuple_list):
        self.tuple_list = tuple_list

    def __len__(self):
        return len(self.tuple_list[0])

    def __getitem__(self, idx):
        return ([item[idx] for item in self.tuple_list])

def get_dataloader(tuple_list, batch_size=32, shuffle=True):
    loader = DataLoader(_RawTensorDataset(tuple_list), batch_size=batch_size, shuffle=shuffle)
    return loader
