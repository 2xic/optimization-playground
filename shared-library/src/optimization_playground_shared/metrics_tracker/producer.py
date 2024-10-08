import torch
import requests
import uuid
from dotenv import load_dotenv
import os
from .metrics import Metrics
import dataclasses
import glob
from typing import List
from .resource_sender import get_cpu_resource_usage, get_ram_resource_usage, get_gpu_resource_usage
import threading
import time
import queue

load_dotenv()

assert "HOST" in os.environ, "Missing host keyword"

class Tracker:
    def __init__(self, project_name) -> None:
        self.hostname = os.environ["HOST"]
        if not "http://" in self.hostname:
            self.hostname = "http://" + self.hostname
        self.name = project_name
        self.run_id = uuid.uuid4()
        self.metric_queue = queue.Queue()
        # we send feedback -> feedback is good
        self.stop_background_thread = threading.Event()
        # sending resource updates :O
        self.background_thread = threading.Thread(target=self.start_background_thread, daemon=True)
        self.background_thread.start()

    def start_background_thread(self):
        print("Starting to send resource usage")
        while not self.stop_background_thread.set() or not self.metric_queue.empty():
            if not self.metric_queue.empty():
                print(f"Cleaning queue {self.metric_queue.qsize()}")
            while not self.metric_queue.empty():
                item = self.metric_queue.get(block=False)
                self.log(item)
            try:
                response = requests.post(self.hostname + "/resource_usage", json={
                    "cpu": get_cpu_resource_usage(),
                    "ram": get_ram_resource_usage(),
                    "gpu": get_gpu_resource_usage(),
                })
                assert response.status_code == 200, "bad status code"
                time.sleep(5)
            except Exception as e:
                print("Exception in background thread", e)
                time.sleep(30)
        print("Done background thread")

    def send_code_state(self, folders: List[str]):
        files = {}
        # we only care about scripts :D 
        for folder_path in folders:
            path = glob.glob(os.path.join(
                os.path.dirname(folder_path),
                "**",
                "*.py"
            )) + glob.glob(os.path.join(
                os.path.dirname(folder_path),
                "*.py"
            ))
            print(path)
            for i in path:
                with open(i, "r") as file:
                    root_name = i.replace(folder_path, "")
                    name = os.path.join(
                        os.path.dirname(folder_path),
                        root_name
                    ) 
                    files[name] = file.read()
        print(requests.post(self.hostname, json={
            "name": self.name,
            "run_id": str(self.run_id),
            "code": files,
            "message_type": "code_state",
        }))
        return self

    def log(self, metrics: Metrics):
        print(requests.post(self.hostname, json={
            "name": self.name,
            "run_id": str(self.run_id),
            "metrics": dataclasses.asdict(metrics),
            "message_type": "metrics",
        }))

    def queue(self, metrics: Metrics):
        self.metric_queue.put(metrics)

    def stop(self):
        self.stop_background_thread.set()


if __name__ == "__main__":
    tracker = Tracker(
        "testing-tracker-v2"
    )
    for epoch in range(10):
        tracker.log(Metrics(
            epoch=epoch,
            loss=torch.rand(1).item()
        ))
        time.sleep(3)
