import cudaplayground
import numpy as np
"""
cudaplayground.pare_array(
    np.zeros((2)).tolist()
)
print("expected == 1")

print("=" * 18)

cudaplayground.pare_array(
    np.zeros((3, 2, 3)).tolist()
)
print("expected == 3")
print("=" * 18)

cudaplayground.pare_array(
    np.zeros((3, 3, 2, 3)).tolist()
)
print("expected == 4")
"""
"""
X = np.random.rand(3, 2)
arr = cudaplayground.pare_array(
    X.tolist()
)
print(X.tolist())
print("expected == 2")
print("=" * 18)
print(arr)
arr.print()
print(
    np.zeros((3, 2))#.tolist()
)
"""

X = np.random.rand(3, 2)
arr = cudaplayground.pare_array(
    X.tolist()
)
print(arr)
W = arr.T()
print(W)
W.print()
arr.print()
