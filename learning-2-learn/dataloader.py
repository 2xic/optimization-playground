from dataset import get_quadratic_function_error, get_random_matrix_vector
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class QuadraticFunctionsDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return get_random_matrix_vector()
