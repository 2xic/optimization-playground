import torch
import requests
import uuid
from dotenv import load_dotenv
import os
from .metrics import Metrics
import dataclasses

load_dotenv()

class Tracker:
    def __init__(self, project_name) -> None:
        self.name = project_name
        self.run_id = uuid.uuid4()

    def log(self, metrics: Metrics):
        print(requests.post(os.environ["HOST"], json={
            "name": self.name,
            "run_id": str(self.run_id),
            "metrics": dataclasses.asdict(metrics)
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

