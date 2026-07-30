import io
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import __main__
import sequence_to_sequence as _s2s
from sequence_to_sequence import _storage, generate_audio, text_to_tokens, TTS_TAG

for _n in dir(_s2s):
    if not _n.startswith("__"):
        setattr(__main__, _n, getattr(_s2s, _n))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tts_eval")

PROBES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today?",
    "She sells sea shells by the sea shore.",
    "Artificial intelligence will change the world.",
    "Please remember to save your work.",
    "It was the best of times, it was the worst of times.",
    "Thank you very much for listening.",
]


def load_model(device):
    box = _storage()
    tag_path = os.path.join("checkpoints", "tags", TTS_TAG, "latest.json")
    if not box._path_exists(tag_path):
        raise SystemExit("no tts-ljspeech checkpoint found")
    path = json.loads(box.load_bytes(tag_path))["path"]
    raw = torch.load(
        io.BytesIO(box.load_bytes(os.path.join(path, "model.pt"))),
        map_location=device, weights_only=False,
    )
    stats = json.loads(box.load_bytes(os.path.join(path, "stats.json")))
    return raw, path, stats


def score(m):
    if m["gen_len"] < 20:
        return "SHORT"
    if m["gen_repeat_ratio"] > 0.5 or m["gen_unique_ratio"] < 0.05:
        return "DEG"
    if m["gen_hit_max"]:
        return "RUN"
    return "OK"


def greedy_codes(model, text, device, n=60):
    tok = text_to_tokens(text, device=device)
    with torch.no_grad():
        out = model.generate(tok, max_len=n, temperature=0)
    return out[0, 1:].tolist()


def conditioning_diff(model, device):
    a1 = greedy_codes(model, PROBES[0], device)
    a2 = greedy_codes(model, PROBES[0], device)
    b = greedy_codes(model, PROBES[1], device)

    def frac_diff(x, y):
        m = min(len(x), len(y))
        if m == 0:
            return 0.0
        return sum(1 for i in range(m) if x[i] != y[i]) / m

    same = frac_diff(a1, a2)
    diff = frac_diff(a1, b)
    print("-" * 70)
    print(f"conditioning: same-text diff={same:.2f}  diff-text diff={diff:.2f}")
    if diff < 0.1:
        print("  -> decoder IGNORES text (same codes for different sentences)")
    elif diff > same + 0.3:
        print("  -> decoder responds to text (good)")
    else:
        print("  -> weak text conditioning")
    return same, diff


def probe_internals(model, device):
    ta = text_to_tokens(PROBES[0], device=device)
    tb = text_to_tokens(PROBES[1], device=device)
    with torch.no_grad():
        ea = model.encode(ta)
        eb = model.encode(tb)
        enc_var_a = float(ea.std().item())
        enc_diff = float((ea.mean(1) - eb.mean(1)).abs().mean().item())
        enc_scale = float(ea.abs().mean().item())

        bos = torch.full((1, 1), model.config.audio_padding_idx + 1, dtype=torch.long, device=device)
        la = model.decode(bos, ea)[0, -1]
        lb = model.decode(bos, eb)[0, -1]
        lz = model.decode(bos, torch.zeros_like(ea))[0, -1]
        logit_diff_text = float((la - lb).abs().mean().item())
        logit_diff_zero = float((la - lz).abs().mean().item())
        arg_a, arg_b, arg_z = int(la.argmax()), int(lb.argmax()), int(lz.argmax())

    print("-" * 70)
    print(f"encoder: scale={enc_scale:.3f} var={enc_var_a:.3f} "
          f"cross-text mean-diff={enc_diff:.4f}")
    if enc_diff < 1e-3 * max(enc_scale, 1e-6):
        print("  -> ENCODER COLLAPSED (same output for different text)")
    print(f"decoder step0 logits: text-diff={logit_diff_text:.3f} "
          f"zero-mem-diff={logit_diff_zero:.3f}  argmax A/B/zero={arg_a}/{arg_b}/{arg_z}")
    if logit_diff_zero < 0.05:
        print("  -> DECODER IGNORES memory (zeroing encoder barely moves logits)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    model, path, stats = load_model(device)
    print(f"{TTS_TAG}  {path}  step={stats.get('steps')}\n" + "-" * 70)

    marks = []
    for i, text in enumerate(PROBES):
        out = os.path.join(OUT_DIR, f"probe_{i}.wav")
        try:
            m = generate_audio(model, text, out, device=device)
        except Exception as e:
            print(f"[ERR] {text}\n      -> {e}")
            marks.append("ERR")
            continue
        mark = score(m)
        marks.append(mark)
        print(
            f"[{mark}] {text}\n"
            f"      len={m['gen_len']} stop={m['gen_did_stop']} "
            f"uniq={m['gen_unique_ratio']:.2f} rep={m['gen_repeat_ratio']:.2f} -> {out}"
        )

    conditioning_diff(model, device)
    probe_internals(model, device)

    n = len(PROBES)
    good = marks.count("OK")
    print("-" * 70)
    print(f"ok={good}/{n}  wavs in {OUT_DIR}")

    if marks.count("ERR") > n // 2 or marks.count("DEG") > n // 2:
        verdict = "BROKEN"
    elif good >= n * 0.6:
        verdict = "GOOD"
    elif good >= n * 0.3:
        verdict = "OK"
    else:
        verdict = "WEAK"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
