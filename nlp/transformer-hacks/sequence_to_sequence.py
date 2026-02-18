import torch
import torch.nn as nn
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
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TTSConfig:
    text_vocab_size: int
    text_padding_idx: int

    audio_vocab_size: int
    audio_padding_idx: int

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
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        x = self.norm2(x + self.dropout(self.ffn(x)))
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
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))

        cross_out, _ = self.cross_attn(
            x, encoder_out, encoder_out, key_padding_mask=encoder_padding_mask
        )
        x = self.norm2(x + self.dropout(cross_out))

        x = self.norm3(x + self.dropout(self.ffn(x)))
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
        x = self.text_embed(text_tokens)
        x = self.text_pos(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, padding_mask=text_padding_mask)

        return self.encoder_norm(x)

    def decode(self, audio_tokens, encoder_out, encoder_padding_mask=None):
        seq_len = audio_tokens.size(1)

        x = self.audio_embed(audio_tokens)
        x = self.audio_pos(x)
        x = self.dropout(x)

        causal_mask = self.causal_mask[:seq_len, :seq_len]

        for layer in self.decoder_layers:
            x = layer(x, encoder_out, causal_mask, encoder_padding_mask)

        x = self.decoder_norm(x)
        return self.output_proj(x)

    def forward(self, text_tokens, audio_tokens, text_padding_mask=None):
        encoder_out = self.encode(text_tokens, text_padding_mask)
        logits = self.decode(audio_tokens, encoder_out, text_padding_mask)
        return logits

    @torch.no_grad()
    def generate(
        self, text_tokens, max_len=2000, temperature=1.0, bos_token=1025, eos_token=1026
    ):
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

    for i in range(0, len(bucketed_iter._cache), batch_size):
        b = bucketed_iter._cache[i : i + batch_size]
        if len(b) == batch_size:
            yield collate_tts_batch(b, text_pad_idx, audio_pad_idx)


def train(model, dataset, epochs=100, device="cuda"):
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=1024)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    dl = dataset.iter(batch_size=1)
    print(f"total_samples: {dataset.total_samples}")
    print(f"num_batches: {dataset.num_batches}")
    print(f"batch_size: {dataset.batch_size}")
    print(f"dl.total_batches: {dl.total_batches}")
    print(dataset.info)

    for epoch in range(epochs):
        total_loss = 0

        dataloader = bucketed_iter(
            dataset.iter(batch_size=1),
            batch_size=32,
            text_pad_idx=dataset.info["training_metadata"]["text_padding_idx"],
            audio_pad_idx=dataset.info["training_metadata"]["audio_padding_idx"],
        )

        for batch in dataloader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # print(f"Epoch {epoch}: {total_loss:.4f}")
        print(f"Epoch {epoch}: {total_loss / (len(bucketed_iter._cache) // 32):.4f}")
        print(
            f"text: {text.shape}, audio: {audio.shape}, text_max: {text.max()}, audio_max: {audio.max()}"
        )
        generate_audio(model, "Hello world.", f"generated/hello_{epoch}.wav")


import os
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

    g2p = G2p()
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", " "]
    arpabet = [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "ER",
        "EY",
        "F",
        "G",
        "HH",
        "IH",
        "IY",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OY",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UW",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
        "AA0",
        "AA1",
        "AA2",
        "AE0",
        "AE1",
        "AE2",
        "AH0",
        "AH1",
        "AH2",
        "AO0",
        "AO1",
        "AO2",
        "AW0",
        "AW1",
        "AW2",
        "AY0",
        "AY1",
        "AY2",
        "EH0",
        "EH1",
        "EH2",
        "ER0",
        "ER1",
        "ER2",
        "EY0",
        "EY1",
        "EY2",
        "IH0",
        "IH1",
        "IH2",
        "IY0",
        "IY1",
        "IY2",
        "OW0",
        "OW1",
        "OW2",
        "OY0",
        "OY1",
        "OY2",
        "UH0",
        "UH1",
        "UH2",
        "UW0",
        "UW1",
        "UW2",
    ]
    punctuation = list(".,!?;:'-\"")
    tokens = special_tokens + arpabet + punctuation
    token_to_id = {t: i for i, t in enumerate(tokens)}

    phonemes = g2p(text)
    ids = [token_to_id["<bos>"]]
    for p in phonemes:
        if p in token_to_id:
            ids.append(token_to_id[p])
        elif p.strip() == "":
            ids.append(token_to_id[" "])
        else:
            ids.append(token_to_id["<unk>"])
    ids.append(token_to_id["<eos>"])

    text_tokens = torch.tensor([ids], device=device)

    model.eval()
    with torch.no_grad():
        audio_tokens = model.generate(
            text_tokens, max_len=max_len, temperature=temperature
        )

    encodec = EncodecModel.encodec_model_24khz()
    encodec.eval()

    codes = audio_tokens[0, 1:]
    eos_mask = codes == 1026
    if eos_mask.any():
        codes = codes[: eos_mask.nonzero()[0, 0]]
    codes = codes.clamp(0, 1023)
    codes = codes.unsqueeze(0).unsqueeze(0).cpu()

    with torch.no_grad():
        audio = encodec.decode([(codes, None)])

    sf.write(output_path, audio[0].cpu().numpy().T, 24000)
    print(f"Saved {output_path}")
    model = model.train()


if __name__ == "__main__":
    dataset = WebDataloader(
        base_url=os.environ["WEB_DATALOADER"],
        dataset_name="ljspeech_tts",
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
        )
    )
    generate_audio(model, "Hello world.", "generated/hello.wav")
    train(
        model,
        dataset,
        epochs=10_000,
    )
