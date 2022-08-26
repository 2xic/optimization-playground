import torch
from Augmentations import Augmentations


class FixMatch:
    def __init__(self) -> None:
        self.augmentation = Augmentations()
        self.threshold = 0.7

    def loss(self, model, X):
        weak = self.augmentation.get_weak_augmentation(X)
        strong = self.augmentation.get_strong_augmentation(X)

        prediction_weak = self.get_psuedo_label(model(weak))
        prediction_strong = model(strong)

        return torch.nn.CrossEntropyLoss()(torch.sigmoid(prediction_strong), torch.sigmoid(prediction_weak))

    def get_psuedo_label(self, prediction_weak: torch.Tensor):
        indices = torch.argmax(prediction_weak, dim=-1)
        rows_prediction = (prediction_weak.gather(1, indices.reshape((-1, 1)))).reshape(-1)
        rows_to_pseudo = torch.where(rows_prediction[:] > self.threshold)
        pseudo_label = indices[torch.where(rows_prediction[:] > self.threshold)]

        if len(rows_to_pseudo):
            prediction_weak[rows_to_pseudo] = 0
            prediction_weak[rows_to_pseudo[0], pseudo_label] = 1
        return prediction_weak


