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


for _ in range(10):
    syn0 = p.tensor((3,4)).rand()
    cpu = syn0 + syn0
    syn0 = syn0.cuda()

    gpu = (syn0 + syn0).host()

    assert cpu.isEqual(gpu)

"""

#bc001168df4fbe56efd7e040fdbf9d0c76e5d4b1

Cpu
Time used : 0.04590940475463867
Gpu
Time used : 3.9883391857147217

Cpu
Time used : 0.09458374977111816
Gpu
Time used : 3.9850833415985107

Cpu
Time used : 0.02886176109313965
Gpu
Time used : 4.080005407333374

Cpu
Time used : 0.07733345031738281
Gpu
Time used : 4.005583763122559
----

# c4eb661d03302018534519493ef7197381d6bc81

Cpu
Time used : 0.0775148868560791
Gpu
Time used : 1.8787658214569092

Cpu
Time used : 0.08047652244567871
Gpu
Time used : 1.8807446956634521

Cpu
Time used : 0.07916951179504395
Gpu
Time used : 1.9371531009674072

---

#
Cpu
Time used : 0.0623171329498291
Gpu
Time used : 0.06160330772399902

Cpu
Time used : 0.07780265808105469
Gpu
Time used : 0.0620572566986084

Cpu
Time used : 0.0851283073425293
Gpu
Time used : 0.060800790786743164


"""
