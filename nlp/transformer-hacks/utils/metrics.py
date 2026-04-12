import warnings

warnings.filterwarnings("ignore", message=".*TripleDES.*")

import atexit
import os
import queue
import threading
import time
import traceback
from contextlib import contextmanager

import psutil
import pynvml
import torch
from dotenv import load_dotenv
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    delete_from_gateway,
    push_to_gateway,
)
from prometheus_client.exposition import basic_auth_handler


class MetricsTracker:
    def __init__(self, run_id, dataset_name=None, rank=0, system_metrics_interval=30, enabled=True):
        load_dotenv()
        self.enabled = enabled and rank == 0
        self.shutdown = False

        if not self.enabled:
            return

        self.server = f"{os.environ['HOST']}:9091"
        self.password = os.environ["GRAFANA_PASSWORD"]
        self.run_id = run_id
        self.dataset_name = dataset_name or "unknown"
        self.registry = CollectorRegistry()
        self.gauges = {}
        self.queue = queue.Queue()

        self._init_gpu_monitoring()

        self.thread = threading.Thread(target=self._push_loop, daemon=True)
        self.thread.start()

        if system_metrics_interval > 0:
            self.system_metrics_interval = system_metrics_interval
            self.system_thread = threading.Thread(
                target=self._system_metrics_loop, daemon=True
            )
            self.system_thread.start()

        atexit.register(self.close)

    def _init_gpu_monitoring(self):
        """Initialize NVML once and cache GPU handles."""
        self.gpu_handles = []
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            self.gpu_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))

    def _get_gauge(self, name):
        if name not in self.gauges:
            if name == "span_seconds":
                labels = ["run_id", "dataset", "span"]
            elif name.startswith("gpu_") and name != "gpu_count":
                labels = ["gpu_id"]
            elif name in (
                "cpu_percent",
                "ram_percent",
                "ram_used_gb",
                "ram_available_gb",
                "gpu_count",
            ):
                labels = []
            else:
                labels = ["run_id", "dataset"]
            self.gauges[name] = Gauge(name, name, labels, registry=self.registry)
        return self.gauges[name]

    def _auth_handler(self, url, method, timeout, headers, data):
        return basic_auth_handler(
            url, method, timeout, headers, data, "pushuser", self.password
        )

    def _push(self, job, grouping_key=None):
        push_to_gateway(
            self.server,
            job=job,
            grouping_key=grouping_key,
            registry=self.registry,
            handler=self._auth_handler,
        )

    def _collect_system_metrics(self):
        """Collect CPU, RAM, and GPU metrics."""
        mem = psutil.virtual_memory()
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": mem.percent,
            "ram_used_gb": mem.used / (1024**3),
            "ram_available_gb": mem.available / (1024**3),
            "gpu_count": len(self.gpu_handles),
        }

        gpu_metrics = []
        for i, handle in enumerate(self.gpu_handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                gpu_metrics.append(
                    {
                        "gpu_id": i,
                        "gpu_utilization_percent": util.gpu,
                        "gpu_memory_percent": (mem_info.used / mem_info.total) * 100,
                        "gpu_memory_used_gb": mem_info.used / (1024**3),
                        "gpu_memory_total_gb": mem_info.total / (1024**3),
                        "gpu_temperature_celsius": temp,
                    }
                )
            except Exception as e:
                print(f"GPU {i} metrics failed: {e}")

        return metrics, gpu_metrics

    def _system_metrics_loop(self):
        """Background loop to collect and push system metrics."""
        while not self.shutdown:
            try:
                metrics, gpu_metrics = self._collect_system_metrics()

                for name, value in metrics.items():
                    self._get_gauge(name).set(value)

                for gpu in gpu_metrics:
                    gpu_id = str(gpu.pop("gpu_id"))
                    for name, value in gpu.items():
                        self._get_gauge(name).labels(gpu_id=gpu_id).set(value)

                self._push("system")
            except Exception as e:
                print(f"System metrics loop failed: {e}")
                traceback.print_exc()

            time.sleep(self.system_metrics_interval)

    def _push_loop(self):
        while not self.shutdown:
            try:
                metrics = self.queue.get(timeout=1)
                for name, value in metrics.items():
                    self._get_gauge(name).labels(
                        run_id=self.run_id, dataset=self.dataset_name
                    ).set(value)
                self._push("training", grouping_key={"run_id": self.run_id})
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Metrics push loop failed: {e}")
                traceback.print_exc()

    def log(self, **metrics):
        if not self.enabled:
            return
        processed = {
            name: value.item() if torch.is_tensor(value) else value
            for name, value in metrics.items()
        }
        self.queue.put(processed)

    def close(self):
        if not self.enabled or self.shutdown:
            return
        self.shutdown = True

        self.thread.join()
        if hasattr(self, "system_thread"):
            self.system_thread.join()

        try:
            delete_from_gateway(
                self.server,
                job="training",
                grouping_key={"run_id": self.run_id},
                handler=self._auth_handler,
            )
            delete_from_gateway(
                self.server,
                job="system",
                handler=self._auth_handler,
            )
        except Exception as e:
            print(f"Failed to delete metrics from pushgateway: {e}")

        pynvml.nvmlShutdown()

    @contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        start = time.time()
        yield
        self._get_gauge("span_seconds").labels(
            run_id=self.run_id, dataset=self.dataset_name, span=name
        ).set(time.time() - start)
