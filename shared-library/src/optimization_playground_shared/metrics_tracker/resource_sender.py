import nvidia_smi
import psutil

def get_gpu_resource_usage():
    devices = get_gpu_usage()
    return sum(devices.values()) / len(devices)

def get_gpu_usage():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    device = {}
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        #util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # print(f"|Device {i}| Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB | gpu-util: {util.gpu/100.0:3.1%} | gpu-mem: {util.memory/100.0:3.1%} |")
        device[i] = mem.used / mem.total * 100
    return device

def get_cpu_resource_usage():
    return psutil.cpu_percent()

def get_ram_resource_usage():
    return psutil.virtual_memory().percent
