import pynvml


def get_best_gpu(model_size_gb, margins_gb=4):
    """Find GPU with most free memory using pynvml"""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        best_gpu = None
        max_free_memory = 0
        minimum_free_memory = model_size_gb + margins_gb

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = info.free / (1024**3)

            print(f"GPU {i}: {free_gb:.2f} GB free")

            if free_gb >= minimum_free_memory and info.free > max_free_memory:
                max_free_memory = info.free
                best_gpu = i

        pynvml.nvmlShutdown()
        return best_gpu

    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return None


# Estimate the model size
def estimate_model_size_gb(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Assume float32 (4 bytes per parameter)
    size_bytes = total_params * 4
    size_gb = size_bytes / (1024**3)
    return size_gb


def estimate_cuda_size(model):
    model_size = estimate_model_size_gb(model)
    training_multiplier = 3
    activation_multiplier = 1.5

    total_size = model_size * training_multiplier * activation_multiplier
    return total_size


if __name__ == "__main__":
    gpu = get_best_gpu()
    if gpu is not None:
        print(f"Best GPU: {gpu}")
    else:
        print("No suitable GPU found.")
