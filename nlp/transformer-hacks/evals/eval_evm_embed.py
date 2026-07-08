import os
import sys
import json
import base64
import random
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = "http://localhost:1259"
DATASET = "evm-cluster-triplet-256-v2"

PIN = {
    "autoparam-evm-cluster-triplet-256-v2-triplet": "checkpoints/2026-07-05/20260705_020043/step_3877",
}

TAGS = [
    "autoparam-evm-cluster-triplet-256-v2",
    "autoparam-evm-cluster-triplet-256-v2-infonce",
    "autoparam-evm-cluster-triplet-256-v2-infonce-uniform",
    "autoparam-evm-cluster-triplet-256-v2-triplet",
    "autoparam-evm-cluster-triplet-256-v2-triplet-uniform",
]


def resolve(tag):
    r = requests.post(f"{BASE}/list", json={"tags": True})
    r.raise_for_status()
    for e in r.json():
        if e["tag"] == tag:
            return e
    raise SystemExit(f"tag not found: {tag}")


def embed(model_path, data, method="mean"):
    r = requests.post(
        f"{BASE}/embedding",
        json={
            "text_base64": base64.b64encode(data).decode(),
            "dataset": DATASET,
            "model_path": model_path,
            "method": method,
            "normalize": True,
        },
    )
    if r.status_code != 200:
        raise SystemExit(f"{r.status_code}: {r.text}")
    return r.json()["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def average_precision(ranked, positives):
    if not positives:
        return None
    hits = 0
    s = 0.0
    for i, ok in enumerate(ranked, 1):
        if ok:
            hits += 1
            s += hits / i
    return s / len(positives)


def evaluate(items, vecs):
    n = len(items)
    sims = [[cosine(vecs[i], vecs[j]) for j in range(n)] for i in range(n)]

    recall1_base = 0
    recall1_group = 0
    aps = []
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: -sims[i][j])
        nn = order[0]
        if items[nn]["base"] == items[i]["base"]:
            recall1_base += 1
        if items[nn]["group"] == items[i]["group"]:
            recall1_group += 1
        pos = [items[j]["base"] == items[i]["base"] for j in order]
        ap = average_precision(pos, [p for p in pos if p])
        if ap is not None:
            aps.append(ap)

    intra = []
    inter = []
    allp = []
    for i in range(n):
        for j in range(i + 1, n):
            allp.append(sims[i][j])
            if items[i]["group"] == items[j]["group"]:
                intra.append(sims[i][j])
            else:
                inter.append(sims[i][j])

    mean = lambda x: sum(x) / len(x) if x else 0.0
    std = lambda x: (sum((v - mean(x)) ** 2 for v in x) / len(x)) ** 0.5 if x else 0.0

    return {
        "recall1_base": recall1_base / n,
        "recall1_group": recall1_group / n,
        "mAP": mean(aps),
        "intra": mean(intra),
        "inter": mean(inter),
        "sep": mean(intra) - mean(inter),
        "aniso": mean(allp),
        "spread": std(allp),
    }


def main():
    try:
        with open(os.path.join(ROOT, "eval_dataset.json")) as f:
            items = json.load(f)
    except FileNotFoundError:
        import eval_build
        items = eval_build.build()
    random.Random(0).shuffle(items)
    print(f"dataset: {len(items)} contracts, {len(set(i['group'] for i in items))} groups\n")

    rows = []
    for tag in TAGS:
        info = resolve(tag)
        info["model_path"] = PIN.get(tag, info["model_path"])
        blobs = [bytes.fromhex(it["hex"]) for it in items]
        vecs = [embed(info["model_path"], b) for b in blobs]
        vecs2 = [embed(info["model_path"], b) for b in blobs]
        drift = max(abs(a - b) for u, w in zip(vecs, vecs2) for a, b in zip(u, w))
        print(f"{tag.split('v2')[-1] or '-base':<16} drift={drift:.0e}  {info['model_path']}")
        m = evaluate(items, vecs)
        m["tag"] = tag.replace("autoparam-evm-cluster-triplet-256-v2", "base")
        rows.append(m)

    def verdict(r):
        if r["spread"] < 0.02:
            return "DEAD"
        if r["recall1_group"] > 0.8 and r["sep"] > 0.1:
            return "GOOD"
        if r["recall1_group"] > 0.6 and r["sep"] > 0.05:
            return "OK"
        return "WEAK"

    for r in rows:
        r["v"] = verdict(r)
    rows.sort(key=lambda r: (r["v"] == "DEAD", -r["recall1_group"], -r["sep"]))
    print(f"{'tag':<20}{'grp@1':>8}{'sep':>8}{'spread':>8}{'':>8}")
    print("-" * 54)
    for r in rows:
        print(f"{r['tag']:<20}{r['recall1_group']:>8.2f}{r['sep']:>8.2f}{r['spread']:>8.2f}   {r['v']}")
    print("-" * 54)
    alive = [r for r in rows if r["v"] != "DEAD"]
    print(f"winner: {alive[0]['tag'] if alive else 'none'}")


if __name__ == "__main__":
    main()
