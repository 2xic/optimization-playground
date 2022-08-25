from re import S
from numpy import outer
import torch


class Loss:
    def fast_sim(self, Z):
        z_norm = Z / Z.norm(dim=1)[:, None]
        z_norm = Z / Z.norm(dim=1)[:, None]
        res = torch.mm(z_norm, z_norm.transpose(0, 1))
        return res

    def slow_sim(self, z, N):
        s = torch.zeros((2* N, 2* N))
        for i in range(2 * N):
            for j in range(2*N):
                results: torch.Tensor = (
                    z[i].T @ z[j]) / (torch.norm(z[i]) * torch.norm(z[j]))
                # ^ this always output a value that is the same
                # most likely something buggy with the implementation I wrote :)
                # print(pairwise_cosine_similarity(z[i].reshape(1, -1), z[j].reshape(1, -1)))
                # or maybe not, I get the same results as pairwise_cosine_similarity
              #  print(results.mean(dim=-1))

                # me testing a more "stable" loss
                #                s[i][j] = torch.relu((z[i] - z[j])).sum(dim=-1)
                s[i][j] = results.mean(dim=-1)

        return s

    def slow_combine(self, z_1, z_2, N):
        z = []
        for i in range(N):
            z.append(z_1[i])
            z.append(z_2[i])

        output =  (torch.stack(z))
        print(output.shape)

        return output

    def fast_combine(self, z_1, z_2):
        z_fast = torch.zeros(z_1.shape[0] * 2, z_1.shape[1])
        z_fast[::2, :] = z_1  
        z_fast[1::2, :] = z_2 

        return z_fast

    def loss(self, z_1, z_2):
        Z = self.fast_combine(z_1, z_2)
        s = self.fast_sim(Z)
        
        temperature = 0.5
        
        def loss(i, j):
            return -torch.log(
                torch.exp(s[i][j] / temperature)
            ) / (
                (torch.sum(
                 torch.exp(s[i] / temperature)
                 # torch.tensor([
                 #    torch.exp(s[i] / temperature) for k in range(N) if k != i
                    # ])
                 )
                 - torch.exp(s[i][i] / temperature)
                 )
            )

        def loss_compute(k): return loss(2 * k - 1, 2*k) + loss(2*k, 2 * k - 1)

        N = z_1.shape[0]
        loss_value = loss_compute(1).type_as(z_1)
        for k in range(2, N):
            loss_value += loss_compute(k).type_as(z_1)
        loss_value *= 1 / 2 * N

        return loss_value
