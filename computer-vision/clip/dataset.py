import os
from collections import defaultdict
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
 #                           std=[0.229, 0.224, 0.225])
])

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '.flickr8'
)

os.makedirs(path, exist_ok=True)

class Flickr8Parser(Dataset):
    def __init__(self) -> None:
        image_caption = defaultdict(list)
        self.dataset = []
        with open(os.path.join(path, "Flicker8k_Dataset/Flickr8k.lemma.token.txt"), "r") as file:
            content = file.read()
            for i in content.split("\n"):
                image_caption = i.split("\t")
                file_name = image_caption[0].split("#")[0]
                if os.path.isfile(self.get_path(file_name)):
                    self.dataset.append(
                        [file_name, image_caption[-1]]
                    )
        self.dataset = self.dataset[:25]
    
    def get_path(self, filename):
        return os.path.join(
                path, "Flicker8k_Dataset", filename
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename, text = self.dataset[idx]
        return (
            data_transform(torchvision.io.read_image(
                self.get_path(filename)
            )),
            text
        )


