import sys
import base64
import requests

BASE = "http://localhost:1259"

EVM_BYTECODE = bytes.fromhex(
    "608060405234801561001057600080fd5b5060043610610036"
    "5760003560e01c80633fa4f2451461003b5780635524107714610059575b"
    "600080fd5b610043610075565b60405161005091906100a2565b"
)

SAMPLES = [
    b"the quick brown fox jumps over the lazy dog",
    b"contract Foo { function bar() public returns (uint) { return 42; } }",
    bytes.fromhex("608060405234801561001057600080fd5b50"),
]


def _hx(s):
    s = "".join(s.split())
    return bytes.fromhex(s[: len(s) - (len(s) % 2)])


CONTRACT_ERC20 = _hx(
    "608060405234801561001057600080fd5b50600436106100575760003560e01c8063"
    "18160ddd1461005c57806370a082311461007a578063a9059cbb146100aa578063"
    "dd62ed3e146100da575b600080fd5b6100646101"
    "0a565b6040516100719190610200"
    "565b60405180910390f35b610094600480360381019061008f9190610250565b610113"
    "565b6040516100a19190610200565b60405180910390f35b6100c46004803603810190"
    "6100bf9190610290565b61015b565b6040516100d19190610200565b60405180910390"
    "f35b60025481565b60006020528060005260406000206000915090505481565b6000"
    "339050826000808373ffffffffffffffffffffffffffffffffffffffff1681526020"
    "0190815260200160002054101561019057600080fd5b"
)

CONTRACT_STORAGE = _hx(
    "6080604052348015600f57600080fd5b506004361060325760003560e01c80632e64"
    "cec11460375780636057361d146051575b600080fd5b603d6069565b604051604891"
    "906091565b60405180910390f35b6067600480360381019060639190"
    "60d1565b6072"
    "565b005b60005481565b8060008190555050565b600081905091905056"
    "5b608b81607e565b82525050565b600060208201905060a4600083018460845b9291"
    "5050565b600080fd5b60b78160"
    "7e565b811460c157600080fd5b50565b600081359050"
    "60d18160ae565b9291505056"
)


def resolve(tag):
    r = requests.post(f"{BASE}/list", json={"tags": True})
    r.raise_for_status()
    for e in r.json():
        if e["tag"] == tag:
            return e
    raise SystemExit(f"tag not found: {tag}")


def dataset_of(model_path):
    import os
    import json
    from utils.checkpoints import StorageBox

    s = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    stats = json.loads(s.load_bytes(os.path.join(model_path, "stats.json")))
    return stats.get("dataset")


def test_embedding(tag, dataset, model_path):
    methods = ("mean", "max", "first", "last", "weighted_decay")
    vecs = {}
    for method in methods:
        r = requests.post(
            f"{BASE}/embedding",
            json={
                "text_base64": base64.b64encode(EVM_BYTECODE).decode(),
                "dataset": dataset,
                "model_path": model_path,
                "method": method,
                "normalize": True,
            },
        )
        if r.status_code != 200:
            raise SystemExit(f"{r.status_code}: {r.text}")
        out = r.json()
        vec = out["embedding"]
        vecs[method] = vec
        print(f"{method:16} dim={len(vec)} chunks={out['num_chunks']} head={vec[:4]}")
    print("\ncosine similarity:")
    print(f"{'':16}" + "".join(f"{m:>16}" for m in methods))
    for a in methods:
        row = "".join(f"{cosine(vecs[a], vecs[b]):16.4f}" for b in methods)
        print(f"{a:16}{row}")


def embed_one(dataset, model_path, data, method="mean"):
    r = requests.post(
        f"{BASE}/embedding",
        json={
            "text_base64": base64.b64encode(data).decode(),
            "dataset": dataset,
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


def test_compare(tag, dataset, model_path):
    shuf = bytearray(CONTRACT_ERC20)
    for i in range(len(shuf) - 1, 0, -1):
        j = (i * 2654435761) % (i + 1)
        shuf[i], shuf[j] = shuf[j], shuf[i]
    items = {
        "erc20": CONTRACT_ERC20,
        "erc20_shuf": bytes(shuf),
        "storage": CONTRACT_STORAGE,
        "text": SAMPLES[0],
        "solsrc": SAMPLES[1],
    }
    vecs = {k: embed_one(dataset, model_path, v) for k, v in items.items()}
    mean = [sum(c) / len(vecs) for c in zip(*vecs.values())]
    cen = {k: [a - b for a, b in zip(v, mean)] for k, v in vecs.items()}

    keys = list(items)
    print("raw cosine (should be ~flat if anisotropic):")
    print(f"{'':10}" + "".join(f"{k:>10}" for k in keys))
    for a in keys:
        print(f"{a:10}" + "".join(f"{cosine(vecs[a], vecs[b]):10.4f}" for b in keys))
    print("\ncentered cosine (real structure for plotting):")
    print(f"{'':10}" + "".join(f"{k:>10}" for k in keys))
    for a in keys:
        print(f"{a:10}" + "".join(f"{cosine(cen[a], cen[b]):+10.4f}" for b in keys))


FEEDBACK_SAMPLES = [
    b"the quick brown fox jumps over the lazy dog",
    b"This product is absolutely terrible and I want a refund.",
    b"Thank you so much, this was incredibly helpful and clear!",
    b"If AI is doing the writing, it should do the reading too.",
]


def test_feedback(tag, dataset, model_path):
    r = requests.post(
        f"{BASE}/classify",
        json={
            "documents_base64": [base64.b64encode(s).decode() for s in FEEDBACK_SAMPLES],
            "dataset": dataset,
            "model_path": model_path,
            "apply_transform": False,
        },
    )
    if r.status_code != 200:
        raise SystemExit(f"{r.status_code}: {r.text}")
    for i, out in enumerate(r.json()):
        print(f"sample {i}: logits={out['logits']} probs={out['probs']}")


def test_keys(tag, dataset, model_path):
    import io
    import json
    import os
    import torch
    from utils.checkpoints import StorageBox
    from training.model import Config, Model

    s = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    cfg = Config.from_json(json.loads(s.load_bytes(os.path.join(model_path, "config.json"))))
    model = Model(cfg)
    w = torch.load(
        io.BytesIO(s.load_bytes(os.path.join(model_path, "model.pt"))),
        map_location="cpu",
    )
    print("saved type:", type(w))
    if isinstance(w, dict):
        print("saved top-level keys:", list(w.keys())[:12])
    saved = set(w.keys()) if isinstance(w, dict) else set()
    expected = set(model.state_dict().keys())
    print("\nsaved sample:")
    for k in list(saved)[:12]:
        print("  ", k)
    print("\nexpected sample:")
    for k in list(expected)[:12]:
        print("  ", k)
    print("\nmissing (expected, not saved):", list(expected - saved)[:12])
    print("unexpected (saved, not expected):", list(saved - expected)[:12])
    print("\ncounts  saved:", len(saved), "expected:", len(expected))


MODES = {
    "embedding": ("autoparam-evm-cluster-triplet-256-v2", test_embedding),
    "compare": ("autoparam-evm-cluster-triplet-256-v2", test_compare),
    "feedback": ("autoparam-feedback-256-v2", test_feedback),
    "keys": ("autoparam-evm-cluster-triplet-256-v2", test_keys),
}

mode = sys.argv[1] if len(sys.argv) > 1 else "embedding"
if mode not in MODES:
    raise SystemExit(f"mode must be one of {list(MODES)}")

TOKENIZER_DATASET = {"feedback": "fineweb-256"}

default_tag, fn = MODES[mode]
tag = sys.argv[2] if len(sys.argv) > 2 else default_tag

info = resolve(tag)
if len(sys.argv) > 3:
    dataset = sys.argv[3]
elif mode in TOKENIZER_DATASET:
    dataset = TOKENIZER_DATASET[mode]
else:
    dataset = dataset_of(info["model_path"])
print("mode:", mode)
print("tag:", tag)
print("run_id:", info["run_id"], "step:", info["step"])
print("model_path:", info["model_path"])
print("dataset:", dataset)
print()
fn(tag, dataset, info["model_path"])
