import torch

def test_normal_output(z, N):
    s = torch.zeros((2*N, 2*N))
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

def test_fast_output(z, N):
    # from https://stackoverflow.com/a/50426321
    
    a_norm = z / z.norm(dim=1)[:, None]
    b_norm = z / z.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return res

z = torch.rand((6, 6))
output_1 = test_normal_output(
    z,
    N=3
)
print(output_1)

output_2 = test_fast_output(
    z,
    N=3
)
print(output_2)

print(torch.allclose(output_1, output_2))


