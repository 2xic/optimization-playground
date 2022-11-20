import cudaplayground

print("Hello :)")
#print(dir(cudaplayground))
a = cudaplayground.Tensor()
b = cudaplayground.Tensor()

a = a.zeros()
b = b.zeros()

print(a.print())
print(b.print())

print("=" * 8)

c  = a + b

print(c.print())
