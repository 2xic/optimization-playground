import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import MultiheadAttention
import torch.optim as optim
from dataclasses import dataclass
from training.model import (
    PositionalEmbeddings,
    PositionalEmbeddingType,
)
from tqdm import tqdm

# from training.layers import MultiheadAttention
from utils.web_dataloader import WebDataloader
import io
import os
import time
from dotenv import load_dotenv

load_dotenv()

import gzip
import json
import math
import uuid
from dataclasses import asdict

from utils.checkpoints import StorageBox, StorageBoxCheckpoint, Stats, TrainingMetadata
from scheduler.cooperative import install_shutdown_handler, shutdown_requested

install_shutdown_handler()

TTS_TAG = "tts-ljspeech-v2"


def _storage() -> StorageBox:
    return StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )


def _ckpt_writer() -> StorageBoxCheckpoint:
    return StorageBoxCheckpoint(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
        run_id=f"tts-ljspeech-{uuid.uuid4().hex[:8]}",
    )


def _make_stats(global_step, epoch, loss_average, dataset_name) -> Stats:
    meta = TrainingMetadata()
    meta.epoch = epoch
    return Stats(
        accuracy_pct=0.0,
        loss_average=float(loss_average),
        runtime_seconds=0,
        steps=global_step,
        dataset=dataset_name,
        metadata=meta,
    )


def _best_path():
    return os.path.join("checkpoints", "tags", TTS_TAG, "best.json")


def _history_path():
    return os.path.join("checkpoints", "tags", TTS_TAG, "history.jsonl")


def _read_best_loss(box):
    if not box._path_exists(_best_path()):
        return None
    try:
        return json.loads(box.load_bytes(_best_path())).get("loss_average")
    except Exception:
        return None


def _write_best(sbx, box, stats):
    tag_data = {
        "run_id": sbx.run_id,
        "step": stats.steps,
        "path": os.path.join(sbx.base_name, f"step_{stats.steps}"),
        "loss_average": float(stats.loss_average),
    }
    box.save_bytes(json.dumps(tag_data, indent=2).encode(), _best_path())


def _append_history(box, stats):
    prev = b""
    if box._path_exists(_history_path()):
        try:
            prev = box.load_bytes(_history_path())
        except Exception:
            prev = b""
    line = json.dumps(stats.to_json()).encode() + b"\n"
    box.save_bytes(prev + line, _history_path())


def _save_checkpoint(sbx, model, optimizer, stats):
    try:
        sbx.checkpoint(model, optimizer, asdict(model.config), stats).result()
    except Exception as e:
        print(f"Checkpoint upload failed at step={stats.steps}, not tagging: {e}", flush=True)
        return
    print(f"Saved checkpoint at step={stats.steps} epoch={stats.metadata.epoch}", flush=True)
    try:
        sbx.tag(TTS_TAG, stats).result()
    except Exception as e:
        print(f"Tag update failed at step={stats.steps}: {e}", flush=True)
        return
    box = _storage()
    try:
        prev_best = _read_best_loss(box)
        if prev_best is None or stats.loss_average < prev_best:
            _write_best(sbx, box, stats)
            print(f"New best loss={stats.loss_average:.4f} (prev={prev_best})", flush=True)
    except Exception as e:
        print(f"Best update failed at step={stats.steps}: {e}", flush=True)
    try:
        _append_history(box, stats)
    except Exception as e:
        print(f"History append failed at step={stats.steps}: {e}", flush=True)


def _save_sample(sbx, local_path, stats):
    with open(local_path, "rb") as f:
        data = f.read()
    step_dir = os.path.join(sbx.base_name, f"step_{stats.steps}")
    sbx.save_bytes(data, os.path.join(step_dir, "sample.wav"))
    print(f"Uploaded sample at step={stats.steps}", flush=True)


def _load_checkpoint(model, optimizer, device):
    if os.environ.get("TTS_FRESH_START") == "1":
        print("TTS_FRESH_START=1; ignoring checkpoint, starting fresh", flush=True)
        return 0, 0
    box = _storage()
    tag_path = os.path.join("checkpoints", "tags", TTS_TAG, "latest.json")
    if not box._path_exists(tag_path):
        print("No existing checkpoint; starting fresh", flush=True)
        return 0, 0
    path = json.loads(box.load_bytes(tag_path))["path"]
    if not box._path_exists(os.path.join(path, "model.pt")):
        print(f"Checkpoint pointer {path} dangling; starting fresh", flush=True)
        return 0, 0
    try:
        raw_model = torch.load(
            io.BytesIO(box.load_bytes(os.path.join(path, "model.pt"))),
            map_location=device, weights_only=False,
        )
        raw_opt = torch.load(
            io.BytesIO(box.load_bytes(os.path.join(path, "optimizer.pt"))),
            map_location=device, weights_only=False,
        )
        stats = json.loads(box.load_bytes(os.path.join(path, "stats.json")))
    except Exception as e:
        print(f"Checkpoint {path} unreadable ({e}); starting fresh", flush=True)
        return 0, 0
    model.load_state_dict(raw_model.state_dict() if isinstance(raw_model, nn.Module) else raw_model)
    optimizer.load_state_dict(raw_opt.state_dict() if hasattr(raw_opt, "state_dict") else raw_opt)
    step = stats.get("steps", 0)
    epoch = stats.get("metadata", {}).get("epoch", 0)
    print(f"Resumed checkpoint at step={step} epoch={epoch}", flush=True)
    return step, epoch


@dataclass
class TTSConfig:
    text_vocab_size: int
    text_padding_idx: int

    audio_vocab_size: int
    audio_padding_idx: int

    audio_num_codebooks: int = 1
    audio_codebook_size: int = 1024

    # Shared
    dim_embeddings: int = 512
    num_attention_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    max_text_len: int = 512
    max_audio_len: int = 2048
    dropout: float = 0.1
    feed_forward_dim: int = 2048


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TTSConfig):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=config.dim_embeddings,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.dim_embeddings)
        self.norm2 = nn.LayerNorm(config.dim_embeddings)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim_embeddings, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.dim_embeddings),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, padding_mask=None):
        a = self.norm1(x)
        attn_out, _ = self.self_attn(a, a, a, key_padding_mask=padding_mask)
        x = x + self.dropout(attn_out)

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: TTSConfig):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=config.dim_embeddings,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        self.cross_attn = MultiheadAttention(
            embed_dim=config.dim_embeddings,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(config.dim_embeddings)
        self.norm2 = nn.LayerNorm(config.dim_embeddings)
        self.norm3 = nn.LayerNorm(config.dim_embeddings)

        self.ffn = nn.Sequential(
            nn.Linear(config.dim_embeddings, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.dim_embeddings),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_out, causal_mask=None, encoder_padding_mask=None):
        a = self.norm1(x)
        attn_out, _ = self.self_attn(a, a, a, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(attn_out)

        c = self.norm2(x)
        cross_out, _ = self.cross_attn(
            c, encoder_out, encoder_out, key_padding_mask=encoder_padding_mask, need_weights=False
        )
        x = x + self.dropout(cross_out)

        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


# TODO: more code can be shared with our transformer code.
class TTSTransformer(nn.Module):
    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        self.text_embed = nn.Embedding(
            config.text_vocab_size,
            config.dim_embeddings,
            padding_idx=config.text_padding_idx,
        )
        self.text_pos = PositionalEmbeddings(
            PositionalEmbeddingType.SINUSOIDAL,
            config.max_text_len,
            config.dim_embeddings,
        )
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.dim_embeddings)

        self.audio_embed = nn.Embedding(
            config.audio_vocab_size,
            config.dim_embeddings,
            padding_idx=config.audio_padding_idx,
        )
        self.audio_pos = PositionalEmbeddings(
            PositionalEmbeddingType.SINUSOIDAL,
            config.max_audio_len,
            config.dim_embeddings,
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(config.dim_embeddings)

        self.output_proj = nn.Linear(config.dim_embeddings, config.audio_vocab_size)

        self.dropout = nn.Dropout(config.dropout)

        self._register_causal_mask(config.max_audio_len)

        self.apply(self._init_weights)

    def _register_causal_mask(self, max_len):
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def encode(self, text_tokens, text_padding_mask=None):
        x = self.text_embed(text_tokens) * (self.config.dim_embeddings ** 0.5)
        x = self.text_pos(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, padding_mask=text_padding_mask)

        return self.encoder_norm(x)

    def decode(self, audio_tokens, encoder_out, encoder_padding_mask=None):
        seq_len = audio_tokens.size(1)

        x = self.audio_embed(audio_tokens) * (self.config.dim_embeddings ** 0.5)
        x = self.audio_pos(x)
        x = self.dropout(x)

        causal_mask = self.causal_mask[:seq_len, :seq_len]

        for layer in self.decoder_layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, encoder_out, causal_mask, encoder_padding_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, encoder_out, causal_mask, encoder_padding_mask)

        x = self.decoder_norm(x)
        return self.output_proj(x)

    def forward(self, text_tokens, audio_tokens, text_padding_mask=None):
        encoder_out = self.encode(text_tokens, text_padding_mask)
        logits = self.decode(audio_tokens, encoder_out, text_padding_mask)
        return logits

    @torch.no_grad()
    def generate(
        self, text_tokens, max_len=2000, temperature=1.0, bos_token=None, eos_token=None
    ):
        if bos_token is None:
            bos_token = self.config.audio_padding_idx + 1
        if eos_token is None:
            eos_token = self.config.audio_padding_idx + 2
        self.eval()
        device = text_tokens.device
        batch_size = text_tokens.size(0)

        encoder_out = self.encode(text_tokens)
        generated = torch.full(
            (batch_size, 1), bos_token, dtype=torch.long, device=device
        )

        for _ in range(max_len):
            logits = self.decode(generated, encoder_out)
            next_logits = logits[:, -1, :] / temperature

            if temperature > 0:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token).all():
                break

        return generated


def collate_tts_batch(batch, text_pad_idx=0, audio_pad_idx=1024):
    text_lens = [len(s["text_tokens"]) for s in batch]
    audio_lens = [len(s["audio_tokens"]) for s in batch]

    max_text_len = max(text_lens)
    max_audio_len = max(audio_lens)
    batch_size = len(batch)

    text_tokens = torch.full((batch_size, max_text_len), text_pad_idx, dtype=torch.long)
    audio_tokens = torch.full(
        (batch_size, max_audio_len), audio_pad_idx, dtype=torch.long
    )

    for i, (sample, text_len, audio_len) in enumerate(
        zip(batch, text_lens, audio_lens)
    ):
        t = sample["text_tokens"]
        a = sample["audio_tokens"]

        if isinstance(t, torch.Tensor):
            text_tokens[i, :text_len] = t
        else:
            text_tokens[i, :text_len] = torch.as_tensor(t, dtype=torch.long)

        if isinstance(a, torch.Tensor):
            audio_tokens[i, :audio_len] = a
        else:
            audio_tokens[i, :audio_len] = torch.as_tensor(a, dtype=torch.long)

    return {"text_tokens": text_tokens, "audio_tokens": audio_tokens}


def bucketed_iter(dataloader, batch_size, text_pad_idx=0, audio_pad_idx=1024):
    if not hasattr(bucketed_iter, "_cache"):
        bucketed_iter._cache = []
        for sample in tqdm(dataloader):
            bucketed_iter._cache.append(
                {
                    "text_tokens": sample["text_tokens"][0],
                    "audio_tokens": sample["audio_tokens"][0],
                }
            )
        bucketed_iter._cache.sort(key=lambda x: len(x["audio_tokens"]))
        print(f"Cached {len(bucketed_iter._cache)} samples")

    batches = [
        bucketed_iter._cache[i : i + batch_size]
        for i in range(0, len(bucketed_iter._cache), batch_size)
        if len(bucketed_iter._cache[i : i + batch_size]) == batch_size
    ]
    for idx in torch.randperm(len(batches)).tolist():
        yield collate_tts_batch(batches[idx], text_pad_idx, audio_pad_idx)


def train(model, dataset, epochs=100, device="cuda"):
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=model.config.audio_padding_idx)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    global_step, start_epoch = _load_checkpoint(model, optimizer, device)
    for g in optimizer.param_groups:
        g["lr"] = 1e-3
        g["initial_lr"] = 1e-3

    warmup_steps = 200
    decay_steps = 20000
    min_lr_ratio = 0.1

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if step >= decay_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=global_step - 1
    )
    sbx = _ckpt_writer()
    dataset_name = getattr(dataset, "name", "ljspeech")

    budget_min = float(os.environ.get("TRAINING_TIME_MINUTES", 0) or 0)
    deadline = time.time() + budget_min * 60 if budget_min > 0 else None
    if deadline:
        print(f"Time budget: {budget_min:.0f} min", flush=True)

    print(f"total_samples: {dataset.total_samples}")
    print(f"total_batches: {dataset.total_batches}")
    print(f"batch_size: {dataset.batch_size}")
    print(dataset.info)

    def _stop():
        return (deadline and time.time() > deadline) or shutdown_requested()

    stopped = False
    checkpoint_interval = int(os.environ.get("TTS_CHECKPOINT_INTERVAL", str(60 * 60)))
    last_checkpoint = time.time()
    for epoch in range(start_epoch, epochs):
        if stopped:
            break
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        total_grad_norm = 0.0
        num_batches = 0

        dataloader = bucketed_iter(
            dataset,
            batch_size=32,
            text_pad_idx=dataset.info["training_metadata"]["text_padding_idx"],
            audio_pad_idx=dataset.info["training_metadata"]["audio_padding_idx"],
        )

        for batch in dataloader:
            if _stop():
                stopped = True
                break
            text = batch["text_tokens"].to(device)
            audio = batch["audio_tokens"].to(device)

            audio_in = audio[:, :-1]
            audio_target = audio[:, 1:]

            text_padding_mask = text == model.config.text_padding_idx
            logits = model(text, audio_in, text_padding_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), audio_target.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                mask = audio_target != model.config.audio_padding_idx
                preds = logits.argmax(dim=-1)
                total_correct += ((preds == audio_target) & mask).sum().item()
                total_tokens += mask.sum().item()

            global_step += 1
            total_loss += loss.item()
            total_grad_norm += float(grad_norm)
            num_batches += 1

        avg = total_loss / max(num_batches, 1)
        token_acc = 100.0 * total_correct / max(total_tokens, 1)
        grad_norm_avg = total_grad_norm / max(num_batches, 1)
        perplexity = math.exp(min(avg, 20))
        print(
            f"Epoch {epoch}: loss={avg:.4f} ppl={perplexity:.2f} "
            f"tok_acc={token_acc:.2f}% grad_norm={grad_norm_avg:.3f}",
            flush=True,
        )
        stats = _make_stats(global_step, epoch if stopped else epoch + 1, avg, dataset_name)
        stats.accuracy_pct = token_acc
        stats.metadata["perplexity"] = perplexity
        stats.metadata["grad_norm"] = grad_norm_avg
        stats.metadata["token_accuracy"] = token_acc
        stats.metadata["learning_rate"] = optimizer.param_groups[0]["lr"]
        stats.metadata["tokens_seen"] = total_tokens
        stats.metadata["batches"] = num_batches
        if stopped or epoch == epochs - 1 or time.time() - last_checkpoint >= checkpoint_interval:
            _save_checkpoint(sbx, model, optimizer, stats)
            last_checkpoint = time.time()

    print(f"Training slot done at step={global_step}", flush=True)


import os
import warnings

warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated.*", category=FutureWarning)

from encodec import EncodecModel
from g2p_en import G2p
import torchaudio
import soundfile as sf


def generate_audio(
    model, text, output_path, device="cuda", max_len=1000, temperature=0.8
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = model.to(device)
    model.eval()

    text_tokens = text_to_tokens(text, device=device)

    model.eval()
    with torch.no_grad():
        audio_tokens = model.generate(
            text_tokens, max_len=max_len, temperature=temperature
        )

    return _decode_audio(audio_tokens, output_path, model.config)


def text_to_tokens(text, device="cpu"):
    char_to_id = {" ": 40, ".": 41, ",": 42, "!": 43, "?": 44, ";": 45,
                  ":": 46, "'": 47, "-": 48, '"': 49}
    for i in range(26):
        char_to_id[chr(ord("a") + i)] = 4 + i
    bos, eos, unk = 1, 2, 3
    ids = [bos]
    for c in text.lower():
        ids.append(char_to_id.get(c, unk))
    ids.append(eos)
    return torch.tensor([ids], device=device)


def _decode_audio(audio_tokens, output_path, config):
    encodec = EncodecModel.encodec_model_24khz()
    encodec.eval()

    nq = config.audio_num_codebooks
    cb = config.audio_codebook_size
    eos_token = config.audio_padding_idx + 2

    toks = audio_tokens[0, 1:]
    eos_mask = toks == eos_token
    did_stop = bool(eos_mask.any())
    if did_stop:
        toks = toks[: eos_mask.nonzero()[0, 0]]
    gen_len = int(toks.numel())
    if gen_len > 0:
        unique_ratio = float(toks.unique().numel()) / gen_len
        repeat_ratio = float((toks[1:] == toks[:-1]).sum().item()) / max(gen_len - 1, 1)
    else:
        unique_ratio = 0.0
        repeat_ratio = 0.0
    gen_metrics = {
        "gen_did_stop": did_stop,
        "gen_len": gen_len,
        "gen_hit_max": not did_stop,
        "gen_unique_ratio": unique_ratio,
        "gen_repeat_ratio": repeat_ratio,
    }

    toks = toks[: gen_len // nq * nq]
    codes = toks.reshape(-1, nq).T
    codes = codes - (torch.arange(nq, device=codes.device)[:, None] * cb)
    codes = codes.clamp(0, cb - 1)
    codes = codes.unsqueeze(0).cpu()

    if os.environ.get("TTS_COARSE_ONLY") == "1":
        codes[:, 1:, :] = 0
    encodec.set_target_bandwidth(nq * 0.75)
    with torch.no_grad():
        audio = encodec.decode([(codes, None)])

    sf.write(output_path, audio[0].cpu().numpy().T, 24000)
    print(f"Saved {output_path} {gen_metrics}")
    return gen_metrics


if __name__ == "__main__":
    dataset = WebDataloader(
        base_url=os.environ["WEB_DATALOADER"],
        dataset_name="ljspeech_tts_v2",
        columns=["text_tokens", "audio_tokens"],
        split="train",
        batch_size=1,
    )
    print(dataset.info)
    model = TTSTransformer(
        TTSConfig(
            text_vocab_size=dataset.info["training_metadata"]["text_vocab_size"],
            text_padding_idx=dataset.info["training_metadata"]["text_padding_idx"],
            audio_vocab_size=dataset.info["training_metadata"]["audio_vocab_size"],
            audio_padding_idx=dataset.info["training_metadata"]["audio_padding_idx"],
            audio_num_codebooks=dataset.info["training_metadata"].get("audio_num_codebooks", 1),
            audio_codebook_size=dataset.info["training_metadata"].get("audio_codebook_size", 1024),
        )
    )
    train(
        model,
        dataset,
        epochs=10_000,
    )
