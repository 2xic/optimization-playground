import torch
"""
body part j_1 -> j_2


if the point is on the limb -> value is V
else 0

l_ck = xj_2 - xj_1
0 <= v 
"""

class ParityFields:
    def function(self, p, p_1, p_2, sigma):
        """
        TODO: write optimized version
        """
        # This stays the same for each keypoint.
        v_single =  ((p_2 - p_1)/torch.norm(p_2 - p_1))
        # This stays the same for each keypoint
        l_ck = torch.norm(p_2 - p_1) 
        # This stays the same for each keypoint
        perpendicular = torch.tensor([
            -v_single[1],
            v_single[0]
        ]) 
        # This stays the same for each origin keypoint
        p_delta = p - p_1
        """
        Need this to be a 2D vector output.
        """
       # print(p_delta.shape)
       # print(v_single.shape)
        full_v = torch.concat([
            v_single.reshape((1, 2)) for _ in range(480)
        ], dim=0)
        full_perpendicular = torch.concat([
            perpendicular.reshape((1, 2)) for _ in range(480)
        ], dim=0)
    #    print(full_perpendicular)

        vp = torch.zeros(p.shape[:2])
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                vp[i, j] = torch.norm(v_single @ p_delta[i, j, :])
        
        v_perpendicular = torch.zeros(p.shape[:2])
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                #print(p_delta[i, :, :].shape)
                #results = torch.norm(full_perpendicular @  p_delta[i, :, :].T, dim=1)#.shape)
                v_perpendicular[i, j] = torch.norm(perpendicular @  p_delta[i, j, :]) 
        print(vp)
        print(v_perpendicular)

        v_p_p_1 = vp #torch.norm(p_delta * v_single, dim=2)
        v_norm_p_p_1 = v_perpendicular # torch.norm(p_delta * perpendicular, dim=2)

        z_xx = torch.zeros((v_p_p_1.shape))
        z_xx[0 <= v_p_p_1] += 1
        z_xx[v_p_p_1 <= l_ck] += 1
        z_xx[v_norm_p_p_1 <= sigma ] += 1
        
        zzzz = torch.zeros(v_p_p_1.shape)
        zzzz[3 <= z_xx] = 255
        return zzzz

    # This looks 
    def unoptimized_function(self, p, p_1, p_2, sigma):
        """
        TODO: write optimized version
        """
        v: torch.Tensor = (p_2 - p_1)/torch.norm(p_2 - p_1)
        l_ck = torch.norm(p_2 - p_1) 

        perpendicular = torch.tensor([
            -v[1],
            v[0]
        ]) 

        # v single vector
        # p_delta single vector
        v_p_p_1 = torch.norm(v.float() @ (p - p_1))

        v_norm_p_p_1 = torch.norm(perpendicular.float() @ (p.float() - p_1))

        if 0 <= v_p_p_1 and v_p_p_1 <= l_ck:
            if torch.norm(v_norm_p_p_1) < sigma:
                return v
        return 0


def trapezoidal(n=10_000):
    dx = 1 / n
    # (1 - u) * dj + udj
    f = lambda u: (1 - u) + u
    results = 0
    for i in range(0, n):
        x_k = dx * i
        x_n = dx * n
        if i > 0 and (i + 1) < n:
            results += 2 * f(x_k)        
        else:
            results +=  f(x_k)
    return dx/2 * results


def E():
    # trapezoidal * 
    # function = L(p(u))
    # p = (1 - u) p_1 + u * p_2
    E = trapezoidal()

if __name__ == "__main__":
    print(trapezoidal())
    """
    from coco import Coco
    import matplotlib.pyplot as plt
    dataset = Coco().get_paf_map(10, optimized=False)
    data = torch.trapezoid(dataset, torch.tensor([0, 1]))
    print(data)
    plt.imshow(dataset)
    plt.show()
    """
    
