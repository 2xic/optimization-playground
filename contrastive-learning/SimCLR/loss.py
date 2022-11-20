import torch
import torch.nn.functional as F
class Loss:
    def __init__(self) -> None:
        self.temperature = 0.5

    def fast_sim(self, Z):
 #       z_norm = Z / Z.norm(dim=1)[:, None]
#        z_norm = Z / Z.norm(dim=1)[:, None]
  #      res = torch.mm(z_norm, z_norm.transpose(0, 1))

        res = F.cosine_similarity(Z.unsqueeze(1), Z.unsqueeze(0), dim=2)
        return res

    def slow_sim(self, z, N):
        s = torch.zeros((2 * N, 2 * N))
        for i in range(2 * N):
            for j in range(2*N):
                results: torch.Tensor = (
                    z[i].T @ z[j]) / (torch.norm(z[i]) * torch.norm(z[j]))
                #results = pairwise_cosine_similarity(z[i].reshape(1, -1), z[j].reshape(1, -1))
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

        output = torch.stack(z)
        #print(output.shape)

        return output

    def fast_combine(self, z_1, z_2):
        z_fast = torch.zeros(z_1.shape[0] * 2, z_1.shape[1])
        z_fast[::2, :] = z_1
        z_fast[1::2, :] = z_2

        return z_fast

    def slow_loss(self, z_1, z_2):
        N = z_1.shape[0]
        Z = self.slow_combine(
            z_1, 
            z_2,
            N
        )     
        s = self.slow_sim(
            Z,
            N
        )   
        def loss(i, j):
            top = torch.exp(s[i][j] / self.temperature)
            bottom = torch.sum(torch.exp(torch.tensor([
                s[i][k] / self.temperature for k in range(s.shape[1]) if k != i
            ])))
            return -torch.log(
                top / bottom
            )

        def loss_compute(k):
            return loss(2 * k - 1, 2*k) + loss(2*k, 2 * k - 1)

        loss_value = 0
        for i in range(1, N):
            loss_value += loss_compute(i) #slow_l_i_j(2 * i - 1, 2 * i) + slow_l_i_j(2 * i, 2 * i - 1)
        return loss_value / (2 * N)

    def loss(self, z_1, z_2):
        z_1 = F.normalize(z_1, dim=1)
        z_2 = F.normalize(z_2, dim=1)

        Z = self.fast_combine(z_1, z_2)
        s = self.fast_sim(Z)

        def loss(i, j):
            top = torch.exp(s[i][j] / self.temperature)

            T = torch.arange(0, s.shape[1])
            T = torch.cat([T[0:i], T[i+1:]])

            bottom = torch.sum(
                torch.exp(s[i][T] / self.temperature)
            )
            return -torch.log(
              top / bottom  
            ) 

        def loss_compute(k):
            return loss(2 * k - 1, 2*k) + loss(2*k, 2 * k - 1)

        N = z_1.shape[0]
        loss_value = 0
        for k in range(1, N):
            loss_value += loss_compute(k).type_as(z_1)
        return loss_value / (2 * N)
        
