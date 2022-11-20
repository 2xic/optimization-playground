import torch

class KeyPoint:
    def __init__(self, x, y, visible) -> None:
        #          https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md
        self.x = x
        self.y = y
        self.visible = visible

    def is_visible(self):
        return self.visible == 2

    @property
    def location(self):
#        return torch.tensor([self.x, self.y]).float()
        return torch.tensor([self.y, self.x]).float()

    def __str__(self):
        return f"({self.x}, {self.y}, {self.visible})"

    def __repr__(self) -> str:
        return self.__str__()
