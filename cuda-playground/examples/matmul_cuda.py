import cudaplayground as p


a = p.tensor((2, 2)).rand().cuda()#.host()
b = p.tensor((2, 2)).rand().cuda()#.host()

c = a.matmul(b)

print("OK")
a.host().print()
b.host().print()
c.host().print()


