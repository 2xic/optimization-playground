import requests

BASE = "http://localhost:1259"
DATASET = "fineweb-256"
TAG = "autoparam-pretrain"

r = requests.post(f"{BASE}/list", json={"tags": True})
r.raise_for_status()
info = next(e for e in r.json() if e["tag"] == TAG)
model_path = info["model_path"]
print(f"tag={TAG}  path={model_path}  run={info.get('run_id')} step={info.get('step')}\n" + "-"*70)

for prompt in ["The capital of France is", "Water boils at", "Once upon a time"]:
    r = requests.post(f"{BASE}/predict", json={
        "documents": [prompt], "dataset": DATASET,
        "model_path": model_path, "apply_transform": False,
    })
    r.raise_for_status()
    out = r.json()[0]
    print(f"PROMPT: {prompt}")
    print(f"  temp:   {out['model_temperature_sampling'][:120]!r}")
    print(f"  argmax: {out['model_argmax_sampling'][:120]!r}\n")
