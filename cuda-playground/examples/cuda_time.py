import time
import cudaplayground as p

def time_it(func):
    start = time.time()

    for _ in range(10_000):
        func()

    time_usage = time.time() - start

    print(f"Time used : {time_usage}")

print("Cpu")
syn0 = p.tensor((3,4)).rand()# - 1
time_it(lambda: (syn0 + syn0)) 

print("Gpu")
syn0 = p.tensor((3,4)).rand().cuda()# - 1
time_it(lambda: (syn0 + syn0)) 

