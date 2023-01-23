from dataset import Flickr8Parser
from torch.utils.data import Dataset, DataLoader
from vocab import Vocab
from model import Model
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms

dataset = Flickr8Parser()
vocab = Vocab()
dataset_loader = DataLoader(dataset,batch_size=32, shuffle=True)

model = Model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

normalize = lambda x:  torch.nn.functional.normalize(x, dim=1)

for i in range(100):
    for index, (x, y) in enumerate(dataset_loader):
        encoded = vocab.encode_sentence(y)
        image ,text, scale = model(
            x,
            encoded
        )

        logits_per_image = scale * image @ text.t()
        logits_per_text = logits_per_image.t()
        
        labels = torch.arange(x.shape[0])
        loss_i = torch.nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_t = torch.nn.CrossEntropyLoss()(logits_per_text, labels)
        loss = (loss_i + loss_t)/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)
"""
Same as in the CLIP example notebook
https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/notebooks/Interacting_with_CLIP.ipynb
"""
with torch.no_grad():
    x = x[:5]
    y = y[:5]
    encoded = encoded[:5]

    image_features, text_features, scale = model(
        x,
        encoded
    )
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T


    descriptions = y
    count = len(descriptions)

    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), y, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(x):
        plt.imshow(transforms.ToPILImage()(image), extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
    plt.savefig('test.png')
    

