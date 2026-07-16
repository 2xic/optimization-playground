import sys, io, os, torch
from dotenv import load_dotenv
load_dotenv()

def load_sd(arg):
    if os.path.isfile(arg):
        return torch.load(arg, map_location="cpu")
    from utils.load_mode_from_checkpoint import load_model_from_path
    m, _ = load_model_from_path(arg)
    return m.state_dict()

def strip(sd):
    if any(k.startswith("base.") for k in sd):
        sd = {k[5:]: v for k, v in sd.items() if k.startswith("base.")}
    return sd

def main():
    if len(sys.argv) < 2:
        print("usage: validate_ckpt.py <model.pt | storage/path>")
        return
    sd = strip(load_sd(sys.argv[1]))
    if "model_state_dict" in sd:
        sd = strip(sd["model_state_dict"])

    fails = []

    emb = sd.get("embeddings.weight")
    out = sd.get("output_layer.weight")
    print(f"embeddings.weight: {'ok' if emb is not None else 'MISSING'}")
    print(f"output_layer.weight: {'ok' if out is not None else 'MISSING'}")
    if emb is not None and out is not None:
        eq = torch.equal(emb, out)
        print(f"tie preserved (values equal): {eq}")
        if not eq:
            fails.append("TIE BROKEN on save -> output head corrupt")
            print(f"  emb norm={emb.float().norm():.3f} out norm={out.float().norm():.3f}")
            print(f"  emb zero%={100*(emb==0).float().mean():.1f} out zero%={100*(out==0).float().mean():.1f}")
    elif emb is None or out is None:
        fails.append("tie key missing")

    for k, v in sd.items():
        if not torch.is_floating_point(v):
            continue
        if not torch.isfinite(v).all():
            fails.append(f"{k}: NaN/Inf")
        elif v.numel() and (v == 0).all():
            fails.append(f"{k}: all-zero")

    dt = {str(v.dtype) for v in sd.values() if torch.is_floating_point(v)}
    print(f"dtypes: {dt}")
    if dt - {"torch.float32"}:
        fails.append(f"non-fp32 weights saved: {dt}")

    print("-" * 50)
    if fails:
        print("VERDICT: FAIL")
        for f in fails:
            print("  -", f)
    else:
        print("VERDICT: PASS (checkpoint round-trips clean)")

if __name__ == "__main__":
    main()
