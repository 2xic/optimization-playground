import torch

class ConfidenceMap:
    def function(self, p, body_parts, sigma):
        # P = matrix
        # body_parts = matrix
        # apply torch norm to individual
        return torch.exp(
            - torch.norm(p - body_parts, dim=2) /
            sigma ** 2
        )

if __name__ == "__main__":
    from coco import Coco
    dataset = Coco().load_annotations().plot_confidence(sigma=5)
