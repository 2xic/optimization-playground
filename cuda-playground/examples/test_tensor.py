import cudaplayground

print("Hello :)")
p = cudaplayground
#print(dir(cudaplayground))
a = p.tensor_f()
c = p.tensor_f()
c.zeros()

#a = a.zeros()
print("a")
a.print()
print("===")

b = p.tensor_f()
b = b.ones()

print("= * 16")

print("b")
b = b.print()
print("a")
a = a.print()
print(a)

print("cccc")
c.print()
c.ones()

print(c)
c.print()
a.print()

#print("b")
#b.print()
print("c")
c.print()

# TODO: Tomorrow
c  = a + b
print(c.print())
c = c + c
print(c.print())

print(b)
print(a)
print(c)
print("ready for peacful shutdown :)")
