from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prometheus_client.exposition import basic_auth_handler
from dotenv import load_dotenv
import os
import threading
import queue
import time
from contextlib import contextmanager


class MetricsTracker:
    def __init__(self, run_id, dataset_name=None, rank=0):
        load_dotenv()
        self.enabled = (rank == 0)

        host = os.environ["HOST"]
        self.password = os.environ["GRAFANA_PASSWORD"]

        self.server = f"{host}:9091"
        self.run_id = run_id
        self.dataset_name = dataset_name or "unknown"
        self.registry = CollectorRegistry()
        self.gauges = {}

        self.queue = queue.Queue()
        self.shutdown = False
        self.thread = threading.Thread(target=self._push_loop, daemon=True)
        self.thread.start()

    def _get_gauge(self, name):
        if name not in self.gauges:
            if name == "span_seconds":
                self.gauges[name] = Gauge(
                    name, name, ["run_id", "dataset", "span"], registry=self.registry
                )
            else:
                self.gauges[name] = Gauge(
                    name, name, ["run_id", "dataset"], registry=self.registry
                )
        return self.gauges[name]

    def _auth_handler(self, url, method, timeout, headers, data):
        return basic_auth_handler(
            url, method, timeout, headers, data, "pushuser", self.password
        )

    def _push_loop(self):
        while not self.shutdown:
            try:
                metrics = self.queue.get(timeout=1)
                for name, value in metrics.items():
                    self._get_gauge(name).labels(
                        run_id=self.run_id, dataset=self.dataset_name
                    ).set(value)
                push_to_gateway(
                    self.server,
                    job="training",
                    registry=self.registry,
                    handler=self._auth_handler,
                )
            except queue.Empty:
                pass

    def log(self, **metrics):
        if not self.enabled:
            return
        self.queue.put(metrics)

    def close(self):
        if not self.enabled:
            return
        self.shutdown = True
        self.thread.join()

    @contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        start = time.time()
        yield
        elapsed = time.time() - start
        self._get_gauge("span_seconds").labels(
            run_id=self.run_id, dataset=self.dataset_name, span=name
        ).set(elapsed)
