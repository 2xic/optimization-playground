import requests

BASE = "http://localhost:1259"
DATASET = "fineweb-256"

PIN = {}

TAGS = [
    "autoparam-finetune",
]

TEMPLATE = "<|user|>\n{q}\n<|end|>\n<|assistant|>\n"

PROBES = [
    ("What is the capital of France?", ["paris"]),
    ("What is 2 + 2?", ["4", "four"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("What language runs in a web browser?", ["javascript"]),
    ("What is the boiling point of water in Celsius?", ["100"]),
    ("Translate the word hello into Spanish.", ["hola"]),
    ("What color do you get mixing blue and yellow?", ["green"]),
    ("Write a Python function that adds two numbers.", ["def", "return"]),
    ("What planet do humans live on?", ["earth"]),
    ("How many days are in a week?", ["7", "seven"]),
]


def resolve(tag):
    r = requests.post(f"{BASE}/list", json={"tags": True})
    r.raise_for_status()
    for e in r.json():
        if e["tag"] == tag:
            return e
    raise SystemExit(f"tag not found: {tag}")


def predict(model_path, prompt):
    r = requests.post(
        f"{BASE}/predict",
        json={"documents": [prompt], "dataset": DATASET, "model_path": model_path, "apply_transform": False},
    )
    if r.status_code != 200:
        raise SystemExit(f"{r.status_code}: {r.text}")
    return r.json()[0]["model_argmax_sampling"]


def answer(raw):
    a = raw.split("<|assistant|>")[-1]
    a = a.split("<|end|>")[0]
    a = a.split("<|user|>")[0]
    return a.strip()


def degenerate(text):
    toks = text.split()
    if len(toks) < 2:
        return True
    if len(set(toks)) <= max(1, len(toks) // 5):
        return True
    return False


def main():
    for tag in TAGS:
        info = resolve(tag)
        model_path = PIN.get(tag, info["model_path"])
        print(f"{tag}  {model_path}\n" + "-" * 70)
        hits = 0
        coherent = 0
        for q, keys in PROBES:
            ans = answer(predict(model_path, TEMPLATE.format(q=q)))
            low = ans.lower()
            hit = any(k.lower() in low for k in keys)
            ok = not degenerate(ans)
            hits += hit
            coherent += ok
            mark = "OK " if hit else ("gib" if ok else "DEG")
            print(f"[{mark}] {q}\n      -> {ans[:100]}")
        n = len(PROBES)
        print("-" * 70)
        print(f"knows={hits}/{n}  coherent={coherent}/{n}")

        def verdict():
            if coherent < n // 2:
                return "BROKEN"
            if hits >= n * 0.6:
                return "GOOD"
            if hits >= n * 0.3:
                return "OK"
            return "PARROT"

        print(f"verdict: {verdict()}\n")


if __name__ == "__main__":
    main()
