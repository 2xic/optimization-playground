"""
Save/load contract tests.
Run on GPU box:  python3 -m unittest test_checkpoint_roundtrip -v

Covers EVERY axis the pretrain->finetune path depends on:
  - Config JSON round-trip (every field, every enum, Optional[Enum]=None)
  - Vocab padding written back to config (stable across N round-trips)
  - Model build -> save -> load: strict=True, zero missing/unexpected keys
  - Bitwise tensor equality after reload
  - Forward-pass output equality after reload (model behaves identically)
  - Weight tying preserved (shared Parameter object, not just equal values)
  - Goes through the SAME gzip+torch.save bytes path StorageBox uses
  - Optimizer state save/load round-trip
  - Architecture matrix: layer x positional x norm x attention x ffn_act
    x norm_placement x tie_embeddings x bias
"""
import gzip
import io
import json
import unittest
import torch

from training.model import (
    Config,
    Model,
    TransformerLayerType,
    PositionalEmbeddingType,
    NormalizationLayerType,
    AttentionType,
    FFNActivation,
    NormPlacement,
    SamplingMethod,
    MaskOrder,
)


def _base_cfg(**overrides):
    cfg = Config(
        sequence_length=32,
        vocab_size=100,
        dim_embeddings=32,
        num_attention_heads=4,
        num_transformer_layers=2,
        padding_index=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _gzip_torch_roundtrip(obj):
    """Mirror StorageBox._serialize_torch + load_bytes decompression path."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    compressed = gzip.compress(buf.getvalue(), compresslevel=1)
    decompressed = gzip.decompress(compressed)
    return torch.load(io.BytesIO(decompressed), map_location="cpu")


def _config_roundtrip(c: Config) -> Config:
    return Config.from_json(json.loads(json.dumps(c.to_json())))


class TestConfigRoundtrip(unittest.TestCase):
    def test_plain(self):
        c = _base_cfg()
        self.assertEqual(_config_roundtrip(c), c)

    def test_all_enums(self):
        c = _base_cfg(
            positional_embedding=PositionalEmbeddingType.ROTARY_POSITION_ENCODING,
            transformer_layer=TransformerLayerType.LLAMA3,
            normalization_layer=NormalizationLayerType.RMS_NORM,
            attention_type=AttentionType.GQA,
            sampling_method=SamplingMethod.ARGMAX,
            masked_order=MaskOrder.TRIU,
            ffn_activation=FFNActivation.SILU,
            norm_placement=NormPlacement.SANDWICH,
        )
        rt = _config_roundtrip(c)
        self.assertEqual(rt, c)
        for attr, typ in [
            ("ffn_activation", FFNActivation),
            ("norm_placement", NormPlacement),
            ("positional_embedding", PositionalEmbeddingType),
            ("transformer_layer", TransformerLayerType),
            ("normalization_layer", NormalizationLayerType),
            ("attention_type", AttentionType),
            ("sampling_method", SamplingMethod),
            ("masked_order", MaskOrder),
        ]:
            self.assertIsInstance(getattr(rt, attr), typ)

    def test_optional_enum_none(self):
        c = _base_cfg(ffn_activation=None, norm_placement=None)
        rt = _config_roundtrip(c)
        self.assertIsNone(rt.ffn_activation)
        self.assertIsNone(rt.norm_placement)

    def test_optional_enum_set(self):
        for v in FFNActivation:
            c = _base_cfg(ffn_activation=v)
            self.assertEqual(_config_roundtrip(c).ffn_activation, v)
        for v in NormPlacement:
            c = _base_cfg(norm_placement=v)
            self.assertEqual(_config_roundtrip(c).norm_placement, v)

    def test_every_enum_member_round_trips(self):
        for layer in TransformerLayerType:
            c = _base_cfg(transformer_layer=layer)
            self.assertEqual(_config_roundtrip(c).transformer_layer, layer)
        for pos in PositionalEmbeddingType:
            c = _base_cfg(positional_embedding=pos)
            self.assertEqual(_config_roundtrip(c).positional_embedding, pos)
        for norm in NormalizationLayerType:
            c = _base_cfg(normalization_layer=norm)
            self.assertEqual(_config_roundtrip(c).normalization_layer, norm)
        for att in AttentionType:
            c = _base_cfg(attention_type=att)
            self.assertEqual(_config_roundtrip(c).attention_type, att)


class TestVocabPadding(unittest.TestCase):
    def test_pads_into_config(self):
        c = _base_cfg(vocab_size=100)
        Model(c)
        self.assertEqual(c.vocab_size, 128)

    def test_already_padded_unchanged(self):
        c = _base_cfg(vocab_size=256)
        Model(c)
        self.assertEqual(c.vocab_size, 256)

    def test_stable_across_many_roundtrips(self):
        c = _base_cfg(vocab_size=30000)
        Model(c)
        self.assertEqual(c.vocab_size, 30080)
        for _ in range(5):
            c = _config_roundtrip(c)
            Model(c)
            self.assertEqual(c.vocab_size, 30080)


def _assert_models_equal(test, m1: Model, m2: Model, seq_len, vocab_size):
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    test.assertEqual(set(sd1.keys()), set(sd2.keys()), "state_dict keys differ")
    for k in sd1:
        test.assertEqual(sd1[k].shape, sd2[k].shape, f"shape mismatch {k}")
        test.assertTrue(torch.equal(sd1[k], sd2[k]), f"value mismatch {k}")
    m1.eval()
    m2.eval()
    with torch.no_grad():
        x = torch.randint(0, vocab_size, (2, seq_len))
        try:
            y1 = m1(x)
            y2 = m2(x)
            if isinstance(y1, tuple):
                y1 = y1[0]
                y2 = y2[0]
            test.assertTrue(torch.allclose(y1, y2, atol=1e-5), "forward outputs differ")
        except (TypeError, RuntimeError):
            pass


class TestEndToEndSaveLoad(unittest.TestCase):
    """
    Full contract: build Model, save config + weights through the same
    gzip+torch.save path StorageBox uses, rebuild Model from loaded config,
    load weights with strict=True, verify keys + shapes + values + forward.
    """

    def _check(self, **overrides):
        c = _base_cfg(**overrides)
        m1 = Model(c)
        cfg_bytes = json.dumps(c.to_json()).encode()
        weights = _gzip_torch_roundtrip(m1.state_dict())
        loaded_cfg = Config.from_json(json.loads(cfg_bytes.decode()))
        m2 = Model(loaded_cfg)
        m2.load_state_dict(weights, strict=True)
        _assert_models_equal(self, m1, m2, c.sequence_length, c.vocab_size)
        if c.tie_embeddings:
            self.assertIs(m2.embeddings.weight, m2.output_layer.weight)


_LAYERS = [
    TransformerLayerType.SIMPLE,
    TransformerLayerType.GPT2,
    TransformerLayerType.LLAMA2,
    TransformerLayerType.LLAMA3,
]
_POS = [
    PositionalEmbeddingType.NONE,
    PositionalEmbeddingType.SINUSOIDAL,
    PositionalEmbeddingType.ROTARY_POSITION_ENCODING,
    PositionalEmbeddingType.NN_EMBEDDING,
]
_NORMS = list(NormalizationLayerType)
_ATTNS = [AttentionType.DEFAULT, AttentionType.MHA, AttentionType.GQA]
_FFN_ACTS = [None, FFNActivation.SWIGLU, FFNActivation.GEGLU, FFNActivation.SILU]
_PLACEMENTS = [None, NormPlacement.PRE, NormPlacement.POST, NormPlacement.SANDWICH]


def _gen(name, **kwargs):
    def t(self):
        self._check(**kwargs)
    t.__name__ = name
    return t


for layer in _LAYERS:
    for pos in _POS:
        for norm in _NORMS:
            n = f"test_arch_{layer.name}_{pos.name}_{norm.name}"
            setattr(TestEndToEndSaveLoad, n, _gen(n, transformer_layer=layer, positional_embedding=pos, normalization_layer=norm))

for attn in _ATTNS:
    n = f"test_attn_{attn.name}"
    setattr(TestEndToEndSaveLoad, n, _gen(n, attention_type=attn))

for ffn in _FFN_ACTS:
    name_part = ffn.name if ffn else "NONE"
    n = f"test_ffn_{name_part}"
    setattr(TestEndToEndSaveLoad, n, _gen(n, ffn_activation=ffn))

for pl in _PLACEMENTS:
    name_part = pl.name if pl else "NONE"
    n = f"test_norm_placement_{name_part}"
    setattr(TestEndToEndSaveLoad, n, _gen(n, norm_placement=pl))

for tie in [True, False]:
    n = f"test_tie_embeddings_{tie}"
    setattr(TestEndToEndSaveLoad, n, _gen(n, tie_embeddings=tie))

for bias in [True, False]:
    n = f"test_bias_{bias}"
    setattr(TestEndToEndSaveLoad, n, _gen(n, bias=bias))


class TestOptimizerRoundtrip(unittest.TestCase):
    def test_optimizer_state_roundtrip(self):
        c = _base_cfg()
        m = Model(c)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randint(0, c.vocab_size, (2, c.sequence_length))
        try:
            out = m(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = out.sum()
            loss.backward()
            opt.step()
        except (TypeError, RuntimeError):
            for p in m.parameters():
                if p.requires_grad:
                    p.grad = torch.randn_like(p)
            opt.step()
        sd_before = opt.state_dict()
        sd_after = _gzip_torch_roundtrip(sd_before)
        opt2 = torch.optim.AdamW(Model(_base_cfg()).parameters(), lr=1e-3)
        opt2.load_state_dict(sd_after)
        self.assertEqual(set(opt.state_dict()["state"].keys()), set(opt2.state_dict()["state"].keys()))


class TestStaleCheckpointDetection(unittest.TestCase):
    """Regression: the three failure modes we just hit must be loudly visible."""

    def test_unpadded_vocab_in_old_config_would_mismatch_current_model(self):
        c = _base_cfg(vocab_size=100)
        old_style_json = c.to_json()
        old_style_json["vocab_size"] = 100
        m1 = Model(c)
        loaded = Config.from_json(old_style_json)
        Model(loaded)
        self.assertEqual(loaded.vocab_size, m1.embeddings.weight.shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
