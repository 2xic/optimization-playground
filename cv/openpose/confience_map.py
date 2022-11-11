import torch

class ConfidenceMap:
    def __init__(self) -> None:
        pass

    def function(self, p, body_parts):
        #p = torch.concat(x, y, dim=1)
        sigma = 5
      #  p = torch.tensor([x, y])
      #  print(p - body_parts)

        # P = matrix
        # body_parts = matrix
        # apply torch norm to induvidual

        return torch.exp(
            - torch.norm(p - body_parts, dim=2) /
            sigma ** 2
        )

if __name__ == "__main__":
    from coco import Coco
    dataset = Coco().load_annotations().plot_confidence()
