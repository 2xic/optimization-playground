import time
import cudaplayground as p

operators = [
   ("Add", lambda x, y: x + y),
   ("Sub", lambda x, y: x - y),
   ("Mul", lambda x, y: x * y),
   ("Div", lambda x, _: x / 4),
   ("Matmul", lambda x, y: x.matmul(y)),
   ("Exp", lambda x, _: x.exp()),
   ("Transpose", lambda x, _: x.T()),
]

def take_time(op):
    start = time.time()
    results = op()
    end = time.time()

    return results, end - start

N = 1_0000

for (name, op) in operators:
    total_cpu = 0
    total_gpu = 0

    # To not have to spend time allocate on device
    syn0 = p.tensor((3,3)).rand() + 1
    syn1 = (syn0 * 1).cuda()

    for _ in range(N):
        cpu, cpu_time = take_time(lambda: op(syn0, syn0))
        gpu, gpu_time = take_time(lambda: op(syn1, syn1))
        gpu = gpu * 1
        gpu.host()

        total_cpu += cpu_time
        total_gpu += gpu_time

        if not cpu.isEqual(gpu):
            print("Not the same")
            cpu.print()
            gpu.print()
            break

    print(name)    
    print(f"Cpu : {total_cpu}")
    print(f"Gpu : {total_gpu}")
    print("")

