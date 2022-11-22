import cudaplayground

print("Hello :)")
p = cudaplayground
#print(dir(cudaplayground))
a = p.tensor_f()
a = p.tensor_f()
#a = a.zeros()
print("a")
a.print()
print("===")

b = p.tensor_f()
b = b.ones()

print("= * 16")

print("b")
b.print()
print("a")
a.print()

#print("b")
#b.print()

print("=" * 8)
print("c")

c  = a + b
print(c.print())
