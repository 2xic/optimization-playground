# based off the code https://zerobone.net/blog/cs/svd-image-compression/
import numpy as np
from PIL import Image as OpenImage
from optimization_playground_shared.plot.Plot import Plot, Image

A = np.asarray(OpenImage.open("compress_me.png"))
R = A[:,:,0] / 0xff
G = A[:,:,1] / 0xff
B = A[:,:,2] / 0xff

R_U, R_S, R_VT = np.linalg.svd(R)
G_U, G_S, G_VT = np.linalg.svd(G)
B_U, B_S, B_VT = np.linalg.svd(B)

print(R_U.shape, R_S.shape, R_VT.shape)
print(A.shape)

relative_rank = 0.2
max_rank = int(relative_rank * min(R.shape[0], R.shape[1]))
print("max rank = %d" % max_rank)  # 144

def read_as_compressed(U, S, VT, k):
    return (U[:,:k] @ np.diag(S[:k])) @ VT[:k]

R_compressed = read_as_compressed(R_U, R_S, R_VT, max_rank)
G_compressed = read_as_compressed(G_U, G_S, G_VT, max_rank)
B_compressed = read_as_compressed(B_U, B_S, B_VT, max_rank)

compressed_float = np.dstack((R_compressed, G_compressed, B_compressed))
compressed = (np.minimum(compressed_float, 1.0) * 0xff).astype(np.uint8)

inference = Plot().plot_image([
    Image(
        image=compressed,
    ),
], f'compressed.png')
