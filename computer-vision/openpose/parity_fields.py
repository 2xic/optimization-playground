import torch
"""
body part j_1 -> j_2

if the point is on the limb -> value is V
else 0

l_ck = xj_2 - xj_1
0 <= v 
"""

class ParityFields:
    def optimized_function(self, x, y, p_1, p_2, sigma, shape):
        v: torch.Tensor = (p_2 - p_1)/torch.norm(p_2 - p_1)
        l_ck = torch.norm(p_2 - p_1) 

        perpendicular = torch.tensor([
            -v[1],
            v[0]
        ])

        distance = v[0].float() * (x - p_1[0]) + (v[1].float() * (y - p_1[1]))
        distance_per = torch.abs(
            perpendicular[0].float() * (x - p_1[0]) + (perpendicular[1].float() * (y - p_1[1]))
        )
        xyz = torch.zeros(shape)
        (x, y) = torch.where(0 <= distance)
        xyz[x, y] += 1

        (x, y) = torch.where(distance <= l_ck)
        xyz[x, y] += 1

        (x, y) = torch.where(distance_per <= sigma)
        xyz[x, y] += 1

        output = torch.zeros(shape + (2, ))
        x, y = torch.where(xyz >= 3)
        output[x, y] = v

        return output

if __name__ == "__main__":
    from coco import Coco
    import matplotlib.pyplot as plt
    dataset = Coco().get_paf_map(10, optimized=False)
    data = torch.trapezoid(dataset, torch.tensor([0, 1]))
    print(data)
    plt.imshow(dataset)
    plt.show()
