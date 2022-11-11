import torch
"""
body part j_1 -> j_2


if the point is on the limb -> value is V
else 0

l_ck = xj_2 - xj_1
0 <= v 
"""

class ParityFields:
    def __init__(self) -> None:
        pass

    def function(self, x, y, p_1, p_2):
        """
        TODO: write optimized version
        """
        p = torch.tensor([x, y])
        v = (p_2 - p_1)/torch.norm(p_2 - p_1)
        l_ck = torch.norm(p_2 - p_1)

        perp = v.clone()
        perp[0] *= -1

        if 0 <= v@ (p - p_1) and v@ (p - p_1) <= l_ck:
            if torch.norm(
                perp * (p - p_1)
            ) < 15:
                return v
        return 0


if __name__ == "__main__":
 #   p_1 = torch.tensor((355, 367)).float()
 #   p_2 = torch.tensor((423, 314)).float()
#    x = ParityFields()
#    for i in range(500):
#        for j in range(500):
#            res = x.function(i, j, p_1, p_2)
#            if torch.is_tensor(res):
    from coco import Coco
    dataset = Coco().load_annotations().plot_paf()
                



