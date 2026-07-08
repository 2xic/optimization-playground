import os, json
from utils.checkpoints import StorageBox
from training.model import Config, Model

MODEL = "checkpoints/2026-07-05/20260705_020043/step_3877"

s = StorageBox(
    host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
    username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
    password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
)
cfg_raw = json.loads(s.load_bytes(os.path.join(MODEL, "config.json")))
cfg = Config.from_json(cfg_raw)
model = Model(cfg)
n = sum(p.numel() for p in model.parameters())
blob = s.load_bytes(os.path.join(MODEL, "model.pt"))

print("dim_embeddings:", cfg.dim_embeddings)
print("heads:", cfg.num_attention_heads)
print("seq_len:", cfg.sequence_length)
print("vocab:", cfg.vocab_size)
print("params:", f"{n:,}", f"({n/1e6:.1f}M)")
print("model.pt on disk:", f"{len(blob)/1e6:.1f} MB")
print("fp32 RAM:", f"{n*4/1e6:.0f} MB")
