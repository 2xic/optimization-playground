import collections
import time
import torch.distributed as dist
import pynvml


def get_true_gpu_utilization(device):
    """Get actual GPU memory usage across all processes"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(
        device if isinstance(device, int) else device.index
    )
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.used / info.total


class AdaptiveBatchSizer:
    def __init__(
        self,
        initial_batch,
        min_batch=8,
        max_batch=512,
        window_size=10,
        target_utilization=0.9,
        safety_margin=0.05,
    ):
        self.current_batch = initial_batch
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_utilization = target_utilization
        self.safety_margin = safety_margin

        self.memory_history = collections.deque(maxlen=window_size)
        self.next_scale_up = time.time() + 30

    @property
    def has_sample(self):
        return len(self.memory_history) >= self.memory_history.maxlen

    def record_step(self, device):
        utilization = get_true_gpu_utilization(device.index)
        self.memory_history.append(utilization)
        #       peak = torch.cuda.max_memory_allocated(device)
        #       total = torch.cuda.get_device_properties(device).total_memory
        #       self.memory_history.append(peak / total)
        #       torch.cuda.reset_peak_memory_stats(device)
        # print(
        #     f"DEBUG: device={device}, peak={peak / 1e9:.2f}GB, total={total / 1e9:.2f}GB, ratio={peak / total:.2%}"
        # )
        return self.has_sample

    def get_batch_size(self, increment=2):
        if not self.has_sample:
            return self.current_batch

        peak_usage = max(self.memory_history)
        avg_usage = sum(self.memory_history) / len(self.memory_history)

        mem_per_sample = avg_usage / self.current_batch
        increment_cost = mem_per_sample * increment

        headroom = self.target_utilization - peak_usage - self.safety_margin

        if headroom > increment_cost * 2 and time.time() > self.next_scale_up:
            rank = dist.get_rank() if dist.is_initialized() else 0
            new_batch = min(self.current_batch + increment, self.max_batch)
            if rank == 0:
                print(
                    f"Scaling {self.current_batch} -> {new_batch} "
                    f"(+{increment_cost * 100:.1f}%, headroom was {headroom * 100:.1f}%)"
                )
            self.current_batch = new_batch
            self.memory_history.clear()
            self.next_scale_up = time.time() + 30

        return self.current_batch
