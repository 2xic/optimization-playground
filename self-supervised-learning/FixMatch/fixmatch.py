import torch
from Augmentations import Augmentations
from Parameters import loss_reduction, output_reduction

class FixMatch:
    def __init__(self) -> None:
        self.augmentation = Augmentations()
        self.adjust_label_size = True
        self.threshold = 0.7

    def loss(self, model, X):
        weak = self.augmentation.get_weak_augmentation(X)
        strong = self.augmentation.get_strong_augmentation(X)

        prediction_weak, rows_kept = self.get_psuedo_label(output_reduction(model(weak).clone().detach()))

        if rows_kept.shape[0] == 0:
            return torch.tensor((0))

        prediction_strong = model(strong)

        assert len(prediction_weak.shape) == 2
        return torch.nn.CrossEntropyLoss(reduction=loss_reduction)(prediction_strong[rows_kept], prediction_weak)

    def get_psuedo_label(self, prediction_weak: torch.Tensor):
        indices = torch.argmax(prediction_weak, dim=-1)
        rows_prediction = (prediction_weak.gather(1, indices.reshape((-1, 1)))).reshape(-1)
        rows_to_pseudo = torch.where(rows_prediction[:] > self.threshold)
        pseudo_label = indices[torch.where(rows_prediction[:] > self.threshold)]

        if len(rows_to_pseudo):
            prediction_weak[rows_to_pseudo] = 0
            prediction_weak[rows_to_pseudo[0], pseudo_label] = 1
            if self.adjust_label_size:
                prediction_weak = prediction_weak[rows_to_pseudo[0], :]
                rows_kept = rows_to_pseudo[0]
            else:
                rows_kept = torch.arange(0, prediction_weak.shape[0])
        return prediction_weak, rows_kept
