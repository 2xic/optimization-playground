import cudaplayground

print("Hello :)")
p = cudaplayground
#print(dir(cudaplayground))
a = p.tensor_f()
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

#print("b")
#b.print()

print("=" * 8)
print("c -> ")
b = b.print()
print(b)
print("c0000")
c = c.print()
print(c)


#print("b")
#b.print()
#print("c")
#c.print()

# TODO: Tomorrow
#c  = a + b
#print(c.print())

print(b)
print(a)
print(c)
