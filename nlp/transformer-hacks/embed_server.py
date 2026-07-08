import os, json, base64, requests
from flask import Flask, request, jsonify

UPSTREAM = "http://localhost:1259/embedding"
UPSTREAM_BATCH = "http://localhost:1259/embedding_batch"
CFG = json.load(open("chosen_model.json"))
SECRET = os.environ.get("EMBED_SECRET")

app = Flask(__name__)


def to_blob(raw):
    if not isinstance(raw, str):
        raise ValueError("not a string")
    raw = raw[2:] if raw.startswith("0x") else raw
    return bytes.fromhex("".join(raw.split()))


def upstream(blob):
    try:
        r = requests.post(UPSTREAM, json={
            "text_base64": base64.b64encode(blob).decode(),
            "dataset": CFG["dataset"],
            "model_path": CFG["model_path"],
            "method": CFG["method"],
            "normalize": CFG["normalize"],
        }, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"upstream unreachable: {e}")
    if r.status_code != 200:
        raise RuntimeError(r.text)
    try:
        return r.json()["embedding"]
    except (ValueError, KeyError):
        raise RuntimeError("bad upstream response")


@app.post("/embed")
def embed():
    if SECRET and request.headers.get("x-secret") != SECRET:
        return jsonify(error="unauthorized"), 401
    data = request.get_json(force=True)
    raw = data.get("bytecode") or data.get("hex")
    if not raw:
        return jsonify(error="send {\"bytecode\": \"<hex>\"}"), 400
    try:
        blob = to_blob(raw)
    except ValueError:
        return jsonify(error="bad hex"), 400
    try:
        vec = upstream(blob)
    except RuntimeError as e:
        return jsonify(error=str(e)), 502
    return jsonify(embedding=vec, model=CFG["model_path"])


@app.post("/embed_batch")
def embed_batch():
    if SECRET and request.headers.get("x-secret") != SECRET:
        return jsonify(error="unauthorized"), 401
    data = request.get_json(force=True)
    items = data.get("bytecodes") or data.get("hexes")
    if not isinstance(items, list) or not items:
        return jsonify(error="send {\"bytecodes\": [\"<hex>\", ...]}"), 400

    docs = []
    bad = {}
    for i, raw in enumerate(items):
        try:
            docs.append((i, base64.b64encode(to_blob(raw)).decode()))
        except ValueError:
            bad[i] = "bad hex"

    out = [{"i": i, "error": bad[i]} for i in bad]
    if docs:
        try:
            r = requests.post(UPSTREAM_BATCH, json={
                "documents_base64": [d for _, d in docs],
                "dataset": CFG["dataset"],
                "model_path": CFG["model_path"],
                "method": CFG["method"],
                "normalize": CFG["normalize"],
            }, timeout=120)
        except requests.RequestException as e:
            return jsonify(error=f"upstream unreachable: {e}"), 502
        if r.status_code != 200:
            return jsonify(error=r.text), 502
        for res in r.json()["results"]:
            orig_i = docs[res["i"]][0]
            res["i"] = orig_i
            out.append(res)

    out.sort(key=lambda x: x["i"])
    return jsonify(results=out, model=CFG["model_path"])


@app.get("/health")
def health():
    return jsonify(ok=True, model=CFG["model_path"])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=1260)
