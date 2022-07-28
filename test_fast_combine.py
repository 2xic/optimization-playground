import torch

z_1 = torch.rand(10, 10)
z_2 = torch.rand(10, 10)

N = 10

z = []
for i in range(N):
    z.append(z_1[i])
    z.append(z_2[i])

print(torch.stack(z))

#print(
#torch.stack([
#    a, b
#], dim=2)
#)



z_fast = torch.zeros(20, 10)
z_fast[::2, :] = z_1  
z_fast[1::2, :] = z_2 
print(z_fast)

assert torch.allclose(torch.stack(z), c)

