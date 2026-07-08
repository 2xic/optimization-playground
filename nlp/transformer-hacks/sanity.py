import base64
import requests

BASE = "http://localhost:1259"
MODEL = "checkpoints/2026-07-04/20260704_195107/step_2382"
DATASET = "evm-cluster-triplet-256-v2"

data = bytes.fromhex("6080604052348015600f57600080fd5b50")
b64 = base64.b64encode(data).decode()


def embed():
    r = requests.post(
        f"{BASE}/embedding",
        json={"text_base64": b64, "dataset": DATASET, "model_path": MODEL, "method": "mean", "normalize": True},
    )
    return r.json()["embedding"]


a, c = embed(), embed()
print("self-cosine:", sum(x * y for x, y in zip(a, c)))
