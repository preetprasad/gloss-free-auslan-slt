"""Fast architecture/parity checks for SignJoey SLT and GASLT modes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn

from config import Config
from slt.reference_model import GlossAttention, ReferencePoseSLT


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def gaslt_attention_parity():
    official_path = (Path(__file__).resolve().parents[1] / "tmp/reference_repos/GASLT"
                     / "signjoey/transformer_layers.py")
    if not official_path.exists():
        print("[skip] cloned GASLT reference is absent; functional tests still run")
        return
    spec = importlib.util.spec_from_file_location("official_gaslt_layers", official_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    torch.manual_seed(7)
    official = module.DeformableMultiHeadedAttention(
        query_type="not", query_nb=3, num_heads=4, size=32, dropout=0.0, num_keys=7)
    local = GlossAttention(heads=4, size=32, dropout=0.0, num_keys=7)
    local.load_state_dict(official.state_dict())
    x = torch.randn(2, 11, 32)
    valid = torch.tensor([[[1] * 11], [[1] * 8 + [0] * 3]], dtype=torch.bool)
    official.eval()
    local.eval()
    expected = official(x, x, x, valid)
    actual = local(x, x, x, valid)
    check(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5),
          "GASLT gloss attention numerically matches the pinned official code")


def _reference_slt_state_name(name: str) -> str:
    """Map pinned SignJoey component names to the local source-compatible names."""
    if name.startswith("sgn."):
        return (name.replace("sgn.ln.", "spatial_embedding.linear.")
                .replace("sgn.norm.", "spatial_embedding.norm."))
    if name == "enc.pe.pe":
        return "source_position.pe"
    if name.startswith("enc.layer_norm."):
        return name.replace("enc.layer_norm.", "encoder_norm.")
    if name.startswith("enc.layers."):
        return (name.replace("enc.layers.", "encoder_layers.")
                .replace(".feed_forward.layer_norm.", ".ff.norm.")
                .replace(".feed_forward.pwff_layer.", ".ff.layers.")
                .replace(".src_src_att.", ".attention.")
                .replace(".layer_norm.", ".norm."))
    if name.startswith("txt."):
        return (name.replace("txt.lut.", "word_embedding.lookup.")
                .replace("txt.norm.", "word_embedding.norm."))
    if name == "dec.pe.pe":
        return "target_position.pe"
    if name.startswith("dec.layer_norm."):
        return name.replace("dec.layer_norm.", "decoder_norm.")
    if name.startswith("dec.output_layer."):
        return name.replace("dec.output_layer.", "generator.")
    if name.startswith("dec.layers."):
        return (name.replace("dec.layers.", "decoder_layers.")
                .replace(".trg_trg_att.", ".self_attention.")
                .replace(".src_trg_att.", ".source_attention.")
                .replace(".x_layer_norm.", ".self_norm.")
                .replace(".dec_layer_norm.", ".source_norm.")
                .replace(".feed_forward.layer_norm.", ".ff.norm.")
                .replace(".feed_forward.pwff_layer.", ".ff.layers."))
    return name


def slt_full_forward_parity():
    """Copy every official tensor and require bit-identical eval logits."""
    reference_root = Path(__file__).resolve().parents[1] / "tmp/reference_repos/SLT"
    if not reference_root.exists():
        print("[skip] cloned SLT reference is absent; functional tests still run")
        return

    # This pinned 2020 code imports torchtext only for a Dataset type in a
    # helper module. Its neural components do not use torchtext in this test.
    try:
        import torchtext.data  # noqa: F401
    except (ImportError, OSError):
        torchtext = types.ModuleType("torchtext")
        torchtext_data = types.ModuleType("torchtext.data")
        torchtext_data.Dataset = object
        torchtext.data = torchtext_data
        sys.modules["torchtext"] = torchtext
        sys.modules["torchtext.data"] = torchtext_data
    sys.path.insert(0, str(reference_root))
    try:
        from signjoey.embeddings import Embeddings, SpatialEmbeddings
        from signjoey.encoders import TransformerEncoder
        from signjoey.decoders import TransformerDecoder
    finally:
        sys.path.pop(0)

    vocab_size, hidden = 37, 512
    official = nn.ModuleDict({
        "sgn": SpatialEmbeddings(
            embedding_dim=hidden, input_size=204, num_heads=8,
            norm_type="batch", activation_type="softsign"),
        "enc": TransformerEncoder(
            hidden_size=hidden, ff_size=2048, num_layers=3, num_heads=8,
            dropout=0.1, emb_dropout=0.1),
        "txt": Embeddings(
            embedding_dim=hidden, num_heads=8, norm_type="batch",
            activation_type="softsign", vocab_size=vocab_size, padding_idx=0),
        "dec": TransformerDecoder(
            num_layers=3, num_heads=8, hidden_size=hidden, ff_size=2048,
            dropout=0.1, emb_dropout=0.1, vocab_size=vocab_size),
    })
    cfg = Config(
        model_arch="signjoey_slt", keypoint_subset="body_hands", d_model=hidden,
        nhead=8, num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=2048, target_embedding_dim=hidden,
        visual_feature_dim=0, position_max_len=5000)
    local = ReferencePoseSLT(204, vocab_size, cfg)
    mapped = {_reference_slt_state_name(k): v for k, v in official.state_dict().items()}
    check(set(mapped) == set(local.state_dict()),
          "SL-Transformer state tensors match the pinned official source")
    local.load_state_dict(mapped)

    official.eval()
    local.eval()
    torch.manual_seed(7)
    src = torch.randn(2, 6, 204)
    src_pad = torch.tensor([[False] * 6, [False] * 4 + [True] * 2])
    src_mask = (~src_pad).unsqueeze(1)
    tgt = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 0]])
    tgt_pad = tgt.eq(0)
    tgt_mask = (~tgt_pad).unsqueeze(1)
    with torch.no_grad():
        embedded_src = official.sgn(src, src_mask)
        memory, _ = official.enc(embedded_src, torch.tensor([6, 4]), src_mask)
        embedded_tgt = official.txt(tgt, tgt_mask)
        expected = official.dec(
            trg_embed=embedded_tgt, encoder_output=memory,
            src_mask=src_mask, trg_mask=tgt_mask)[0]
        actual = local(src, src_pad, tgt, tgt_pad)
    check(torch.equal(actual, expected),
          "full SL-Transformer forward is bit-identical to the pinned official source")


def model_check(architecture: str, device: torch.device):
    layers = 2 if architecture == "gaslt" else 3
    dropout = 0.5 if architecture == "gaslt" else 0.1
    target_dim = 300 if architecture == "gaslt" else 512
    cfg = Config(
        model_arch=architecture, visual_feature_dim=1024, d_model=512,
        nhead=8, num_encoder_layers=layers, num_decoder_layers=layers,
        dim_feedforward=2048, dropout=dropout, embedding_dropout=0.1,
        target_embedding_dim=target_dim,
    )
    model = ReferencePoseSLT(204, 3064, cfg).to(device)
    src = torch.randn(2, 17, 204, device=device)
    src_pad = torch.zeros(2, 17, dtype=torch.bool, device=device)
    src_pad[1, 13:] = True
    tgt = torch.tensor([[1, 4, 5, 2, 0], [1, 8, 2, 0, 0]], device=device)
    logits, memory = model(src, src_pad, tgt[:, :-1], tgt[:, :-1].eq(0),
                           return_memory=True)
    check(logits.shape == (2, 4, 3064), f"{architecture} forward shape on {device.type}")
    check(memory.shape == (2, 17, 512), f"{architecture} encoder shape on {device.type}")
    loss = logits.square().mean()
    loss.backward()
    check(torch.isfinite(loss).item(), f"{architecture} backward is finite on {device.type}")
    count = sum(p.numel() for p in model.parameters())
    print(f"[info] {architecture}: {count:,} parameters ({count / 1e6:.2f}M)")


def main():
    slt_full_forward_parity()
    gaslt_attention_parity()
    model_check("signjoey_slt", torch.device("cpu"))
    model_check("gaslt", torch.device("cpu"))
    if torch.backends.mps.is_available():
        model_check("signjoey_slt", torch.device("mps"))
        model_check("gaslt", torch.device("mps"))
    print("REFERENCE ARCHITECTURE TESTS PASSED")


if __name__ == "__main__":
    main()
