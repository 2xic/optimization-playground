import sys, os, torch
from dotenv import load_dotenv

load_dotenv()
from utils.load_mode_from_checkpoint import load_model_from_path
from utils.web_dataloader import WebDataloader

path = sys.argv[1]
args = [a for a in sys.argv[2:] if not a.startswith("--")]
dataset = args[0] if args else "fineweb-256"

model, cfg = load_model_from_path(path)
model.eval()
dl = WebDataloader(os.environ["WEB_DATALOADER"], dataset, batch_size=64)

if "--acc" in sys.argv:
    correct = total = 0
    for i, batch in enumerate(dl):
        X, y = batch["x_tokens"], batch["y_tokens"]
        with torch.no_grad():
            logits = model(X)
        pred = logits.argmax(-1)
        mask = y != cfg.padding_index
        correct += ((pred == y) & mask).sum().item()
        total += mask.sum().item()
        if i >= 20:
            break
    print(f"samples={total}  next-token accuracy: {100 * correct / max(total, 1):.2f}%")

from optimization_playground_shared.nlp.utils.sampling import argmax_sampling
import base64

prompt = "The capital of France is"
r = dl.session.post(
    f"{dl.base_url}/datasets/{dl.dataset_name}/tokenize",
    json={
        "documents_base64": [base64.b64encode(prompt.encode()).decode()],
        "apply_transform": False,
        "add_special_tokens": False,
    },
)
print("tokenize status:", r.status_code, "body:", r.text[:300])
padded = dl.tokenize([prompt.encode()], apply_transform=False, add_special_tokens=False)[0][0]
unpadded = dl.tokenize([prompt.encode()], padding=False, apply_transform=False, add_special_tokens=False)[0][0]
print("pad_idx:", cfg.padding_index, "padded len:", len(padded), "unpadded len:", len(unpadded))
print("PADDED   (serve path):", dl.detokenize(model.generate(padded, 40, argmax_sampling)))
print("UNPADDED (fixed):     ", dl.detokenize(model.generate(unpadded, 40, argmax_sampling)))
