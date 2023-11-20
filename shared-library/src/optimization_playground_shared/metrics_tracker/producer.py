import torch
import requests
import uuid
from dotenv import load_dotenv
import os
from .metrics import Metrics
import dataclasses
import glob
from typing import List
from .resource_sender import get_cpu_resource_usage
import threading
import time

load_dotenv()

assert "HOST" in os.environ, "Missing host keyword"

class Tracker:
    def __init__(self, project_name) -> None:
        self.name = project_name
        self.run_id = uuid.uuid4()
        # sending resource updates :O
        t1 = threading.Thread(target=self.start_background_thread)
        t1.start()
        # we send feedback -> feedback is good

    def start_background_thread(self):
        while True:
            print(requests.post(os.environ["HOST"] + "/resource_usage", json={
                "cpu": get_cpu_resource_usage()
            }))
            time.sleep(30)

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
        print(requests.post(os.environ["HOST"], json={
            "name": self.name,
            "run_id": str(self.run_id),
            "code": files,
            "message_type": "code_state",
        }))
        return self

    def log(self, metrics: Metrics):
        print(requests.post(os.environ["HOST"], json={
            "name": self.name,
            "run_id": str(self.run_id),
            "metrics": dataclasses.asdict(metrics),
            "message_type": "metrics",
        }))

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
