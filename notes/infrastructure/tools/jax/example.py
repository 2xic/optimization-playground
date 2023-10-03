"""
Jax
"""
import jax

def f(x):
  return x**2

x = float(2)
jax_grad_f = jax.grad(f)
dy = jax_grad_f(x)
print(dy)

"""
Torch equivalent
"""
import torch
def f(x):
  return x**2
x = torch.tensor(2.0, requires_grad=True)
y = f(x)
y.backward()
print(x.grad)
