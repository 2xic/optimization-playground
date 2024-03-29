import nvidia_smi
import threading
import random
import psutil

def get_gpu_resource_usage():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    total_gpu_ram = 0
    total_gpu_ram_usage = 0
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        #util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # print(f"|Device {i}| Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB | gpu-util: {util.gpu/100.0:3.1%} | gpu-mem: {util.memory/100.0:3.1%} |")
        total_gpu_ram += mem.total
        total_gpu_ram_usage += mem.used
    return total_gpu_ram_usage / total_gpu_ram

def get_cpu_resource_usage():
    return psutil.cpu_percent()

def get_ram_resource_usage():
    return psutil.virtual_memory().percent
