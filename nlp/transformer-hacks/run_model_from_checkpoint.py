import warnings

warnings.filterwarnings("ignore", message=".*TripleDES.*")

from utils.checkpoints import StorageBox
import os
import base64
import binascii
from dotenv import load_dotenv
import json
import logging
from typing import Optional
from dataclasses import dataclass
import torch
from utils.web_dataloader import WebDataloader
from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.exceptions import HTTPException
from utils.load_mode_from_checkpoint import (
    load_best_model_from_checkpoint,
    load_model_from_path,
    load_head_model_from_path,
)
from functools import lru_cache
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    argmax_sampling,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

import flask.cli

flask.cli.show_server_banner = lambda *a, **k: None

app = Flask(__name__)


class BadRequest(Exception):
    pass


@dataclass
class BestModelResult:
    loss: Optional[int] = None
    accuracy: Optional[int] = None
    path: Optional[str] = None

    def update_by_loss(self, loss, path):
        if self.loss is None or loss < self.loss:
            self.loss = loss
            self.path = path

    def update_by_accuracy(self, accuracy, path):
        if self.accuracy < accuracy:
            self.accuracy = accuracy
            self.path = path


def get_json():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("body must be a JSON object")
    return data


def require(data, key):
    if key not in data:
        raise BadRequest(f"missing required field: {key}")
    return data[key]


def decode_b64(value):
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError, TypeError):
        raise BadRequest("invalid base64 input")


def parse_bytes(data, base64_key, text_key):
    if base64_key in data:
        return decode_b64(data[base64_key])
    if text_key in data:
        value = data[text_key]
        if not isinstance(value, str):
            raise BadRequest(f"{text_key} must be a string")
        return value.encode()
    raise BadRequest(f"missing required field: {base64_key} or {text_key}")


def parse_documents(data):
    if "documents_base64" in data:
        raw = data["documents_base64"]
        if not isinstance(raw, list):
            raise BadRequest("documents_base64 must be a list")
        return [decode_b64(d) for d in raw]
    if "documents" in data:
        raw = data["documents"]
        if not isinstance(raw, list):
            raise BadRequest("documents must be a list")
        if not all(isinstance(d, str) for d in raw):
            raise BadRequest("documents must be a list of strings")
        return [d.encode() for d in raw]
    raise BadRequest("missing required field: documents_base64 or documents")


@app.errorhandler(BadRequest)
def _handle_bad_request(e):
    return jsonify({"error": str(e)}), 400


@app.errorhandler(Exception)
def _handle_error(e):
    if isinstance(e, HTTPException):
        return e
    logger.exception("unhandled error")
    return jsonify({"error": "internal error"}), 500


@lru_cache(maxsize=4)
def load_model_and_dataloader(
    target_dataset, model_path=None, dataloader_dataset=None, max_age_days=None
):
    if dataloader_dataset is None:
        dataloader_dataset = target_dataset
    if model_path is None:
        model, _ = load_best_model_from_checkpoint(
            target_dataset=target_dataset, max_age_days=max_age_days
        )
    else:
        model, _ = load_model_from_path(model_path)
    dataloader = WebDataloader(
        os.environ["WEB_DATALOADER"],
        dataloader_dataset,
        batch_size=1024,
    )
    return model, dataloader


@lru_cache(maxsize=4)
def load_head_model_and_dataloader(
    target_dataset, model_path, dataloader_dataset=None, num_classes=1
):
    if dataloader_dataset is None:
        dataloader_dataset = target_dataset
    model, _ = load_head_model_from_path(model_path, num_classes=num_classes)
    dataloader = WebDataloader(
        os.environ["WEB_DATALOADER"],
        dataloader_dataset,
        batch_size=1024,
    )
    return model, dataloader


def stream_response(steps):
    def generate():
        try:
            for status in steps:
                if isinstance(status, dict) and "result" in status:
                    yield json.dumps(status) + "\n"
                else:
                    yield json.dumps({"status": status}) + "\n"
        except BadRequest as e:
            yield json.dumps({"error": str(e)}) + "\n"
        except Exception:
            logger.exception("stream error")
            yield json.dumps({"error": "internal error"}) + "\n"

    return Response(
        stream_with_context(generate()), mimetype="application/x-ndjson"
    )


def pool_embeddings(embeddings, method):
    if method == "mean":
        return torch.mean(embeddings, dim=0)
    if method == "max":
        return torch.max(embeddings, dim=0).values
    if method == "first":
        return embeddings[0]
    if method == "last":
        return embeddings[-1]
    weights = torch.arange(len(embeddings), 0, -1, dtype=torch.float)
    weights = weights / weights.sum()
    return (embeddings * weights.unsqueeze(1)).sum(dim=0)


def embed_one(model, dataloader, text, method, normalize, apply_transform, add_special_tokens):
    doc_tensors = dataloader.tokenize(
        [text],
        apply_transform=apply_transform,
        add_special_tokens=add_special_tokens,
    )
    if not doc_tensors or not len(doc_tensors[0]):
        raise BadRequest("input tokenized to empty sequence")
    with torch.no_grad():
        embeddings = torch.concat([model.embed(v) for v in doc_tensors], dim=0)
    pooled = pool_embeddings(embeddings, method)
    if normalize:
        pooled = torch.nn.functional.normalize(pooled, dim=0)
    return pooled, len(doc_tensors)


@app.route("/embedding", methods=["POST"])
def embedding():
    data = get_json()
    text = parse_bytes(data, "text_base64", "text")
    dataset = require(data, "dataset")
    method = data.get("method", "mean")
    normalize = data.get("normalize", False)
    max_age_days = data.get("max_age_days", 3)
    stream = data.get("stream", False)

    if method not in ("mean", "max", "first", "last", "weighted_decay"):
        raise BadRequest(f"unknown method: {method}")

    def work():
        yield "loading model"
        model, dataloader = load_model_and_dataloader(
            dataset, data.get("model_path"), max_age_days=max_age_days
        )
        yield "tokenizing"
        model.eval()
        yield "embedding"
        pooled, num_chunks = embed_one(
            model, dataloader, text, method, normalize,
            data.get("apply_transform", True),
            data.get("add_special_tokens", False),
        )
        yield {
            "result": {
                "embedding": pooled.tolist(),
                "method": method,
                "normalized": normalize,
                "num_chunks": num_chunks,
            }
        }

    if stream:
        return stream_response(work())
    result = None
    for item in work():
        if isinstance(item, dict) and "result" in item:
            result = item["result"]
    return jsonify(result)


@app.route("/embedding_batch", methods=["POST"])
def embedding_batch():
    data = get_json()
    texts = parse_documents(data)
    dataset = require(data, "dataset")
    method = data.get("method", "mean")
    normalize = data.get("normalize", False)
    max_age_days = data.get("max_age_days", 3)

    if method not in ("mean", "max", "first", "last", "weighted_decay"):
        raise BadRequest(f"unknown method: {method}")

    model, dataloader = load_model_and_dataloader(
        dataset, data.get("model_path"), max_age_days=max_age_days
    )
    model.eval()
    apply_transform = data.get("apply_transform", True)
    add_special_tokens = data.get("add_special_tokens", False)

    results = []
    for i, text in enumerate(texts):
        try:
            pooled, num_chunks = embed_one(
                model, dataloader, text, method, normalize,
                apply_transform, add_special_tokens,
            )
            results.append({"i": i, "embedding": pooled.tolist(), "num_chunks": num_chunks})
        except BadRequest as e:
            results.append({"i": i, "error": str(e)})

    return jsonify({"results": results, "method": method, "normalized": normalize})


@app.route("/predict", methods=["POST"])
def predict():
    data = get_json()
    documents = parse_documents(data)
    dataset = require(data, "dataset")
    dataloader_dataset = data.get("dataloader_dataset")
    apply_transform = data.get("apply_transform", True)
    add_special_tokens = data.get("add_special_tokens", False)
    stream = data.get("stream", False)

    def work():
        yield "loading model"
        model, dataloader = load_model_and_dataloader(
            dataset, data.get("model_path"), dataloader_dataset
        )
        model_response = []
        for i, text in enumerate(documents):
            yield f"tokenizing {i + 1}/{len(documents)}"
            tokenized = dataloader.tokenize(
                [text],
                padding=False,
                apply_transform=apply_transform,
                add_special_tokens=add_special_tokens,
            )
            if not tokenized or not len(tokenized[0]):
                raise BadRequest("input tokenized to empty sequence")
            doc_tensors = tokenized[0][0]
            yield f"generating {i + 1}/{len(documents)}"
            model_temperature_sampling = model.generate(
                doc_tensors, 128, temperature_sampling
            )
            model_argmax_sampling = model.generate(doc_tensors, 128, argmax_sampling)
            model_response.append(
            {
                "model_temperature_sampling": dataloader.detokenize(
                    model_temperature_sampling
                ),
                "model_argmax_sampling": dataloader.detokenize(model_argmax_sampling),
            }
        )
        yield {"result": model_response}

    if stream:
        return stream_response(work())
    result = None
    for item in work():
        if isinstance(item, dict) and "result" in item:
            result = item["result"]
    return jsonify(result)


@app.route("/classify", methods=["POST"])
def classify():
    data = get_json()
    documents = parse_documents(data)
    dataset = require(data, "dataset")
    model_path = require(data, "model_path")
    dataloader_dataset = data.get("dataloader_dataset")
    num_classes = data.get("num_classes", 1)
    apply_transform = data.get("apply_transform", True)
    add_special_tokens = data.get("add_special_tokens", False)
    stream = data.get("stream", False)

    def work():
        yield "loading model"
        model, dataloader = load_head_model_and_dataloader(
            dataset, model_path, dataloader_dataset, num_classes
        )
        model_response = []
        for i, text in enumerate(documents):
            yield f"tokenizing {i + 1}/{len(documents)}"
            tokenized = dataloader.tokenize(
                [text],
                padding=False,
                apply_transform=apply_transform,
                add_special_tokens=add_special_tokens,
            )
            if not tokenized or not len(tokenized[0]):
                raise BadRequest("input tokenized to empty sequence")
            chunks = tokenized[0]
            yield f"predicting {i + 1}/{len(documents)}"
            with torch.no_grad():
                logits = model(chunks)
                if num_classes == 1:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=-1)
            model_response.append(
                {
                    "logits": logits.tolist(),
                    "probs": probs.tolist(),
                }
            )
        yield {"result": model_response}

    if stream:
        return stream_response(work())
    result = None
    for item in work():
        if isinstance(item, dict) and "result" in item:
            result = item["result"]
    return jsonify(result)


@app.route("/list", methods=["POST"])
def list_models():
    data = get_json()
    max_age_days = data.get("max_age_days", 3)
    min_age_days = data.get("min_age_days", 0)
    target_dataset = data.get("target_dataset")
    stream = data.get("stream", False)
    metadata = data.get("metadata", True)
    tags = data.get("tags", False)

    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )

    def work_tags():
        yield "listing tags"
        model_response = []
        for tag_path in storage.list("checkpoints/tags"):
            tag = os.path.basename(tag_path)
            try:
                info = json.loads(
                    storage.load_bytes(os.path.join(tag_path, "best.json"))
                )
            except Exception:
                continue
            pointer = info.get("pointer", info)
            entry = {
                "tag": tag,
                "run_id": pointer.get("run_id"),
                "step": pointer.get("step"),
                "model_path": pointer.get("path"),
            }
            model_response.append(entry)
            yield f"found tag {tag} (run {entry['run_id']} step {entry['step']})"
        model_response = sorted(model_response, key=lambda x: x["tag"])
        yield {"result": model_response}

    def work():
        yield "scanning storage"
        model_response = []
        for filepath in storage.walk(
            max_age_days=max_age_days, min_age_days=min_age_days
        ):
            if os.path.basename(filepath) != "stats.json":
                continue
            model_path = os.path.dirname(filepath)
            try:
                run_id = int(os.path.basename(os.path.dirname(model_path)))
            except ValueError:
                logger.warning("skipping non-integer run id for %s", filepath)
                continue

            if not metadata:
                entry = {
                    "model_path": model_path,
                    "run_id": run_id,
                    "step_dir": os.path.basename(model_path),
                }
                model_response.append(entry)
                yield f"found run {run_id} ({entry['step_dir']})"
                continue

            try:
                stats = json.loads(storage.load_bytes(filepath))
            except Exception as e:
                logger.warning("failed to load %s: %s", filepath, e)
                continue
            if target_dataset is not None and stats.get("dataset") != target_dataset:
                continue
            entry = {
                "model_path": model_path,
                "accuracy_pct": stats.get("accuracy_pct"),
                "dataset": stats.get("dataset"),
                "steps": stats.get("steps"),
                "run_id": run_id,
            }
            model_response.append(entry)
            yield f"found {entry['dataset']} run {run_id} step {entry['steps']}"
        if metadata:
            model_response = sorted(
                model_response, key=lambda x: (x["run_id"], x["steps"])
            )
        else:
            model_response = sorted(
                model_response, key=lambda x: (x["run_id"], x["step_dir"])
            )
        yield {"result": model_response}

    steps = work_tags() if tags else work()
    if stream:
        return stream_response(steps)
    result = None
    for item in steps:
        if isinstance(item, dict) and "result" in item:
            result = item["result"]
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=1259)
