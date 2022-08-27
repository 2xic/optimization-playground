
from Augmentations import Augmentations
from Dataloader import Cifar10Dataloader


aug = Augmentations()
dataloader = Cifar10Dataloader()

(X, y, z) = dataloader[0]

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

pil = F.to_pil_image(X)

plt.figure()

f, axarr = plt.subplots(1, 3) 

axarr[0].imshow(pil)
axarr[1].imshow(F.to_pil_image(aug.get_weak_augmentation(X)))
axarr[2].imshow(F.to_pil_image(aug.get_strong_augmentation(X)))

plt.show()

