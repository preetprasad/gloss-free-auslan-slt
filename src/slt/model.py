"""Baseline model: pose-sequence Transformer encoder-decoder for gloss-free SLT.

Pipeline:  (T, F) normalised HALPE-136 features
           -> optional learned visual feature adapter
           -> linear projection to d_model
           -> sinusoidal positional encoding
           -> Transformer encoder (signer pose context)
           -> Transformer decoder (autoregressive English tokens)
           -> linear generator over the translation vocabulary.

Deliberately small and standard — this is the baseline to beat, not the novel
Phase 3 system. `batch_first=True` throughout.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import BOS, EOS, PAD


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def _coco_wholebody_133_adjacency() -> torch.Tensor:
    """Anatomical skeleton adjacency for the 133-kpt COCO-WholeBody layout used by
    DWPose/RTMW (17 body + 6 feet + 68 face + 21 left hand + 21 right hand — see
    CWB_GROUPS in config.py). Gives PoseGraph's learned adjacency a real skeletal
    prior (true joint neighbours) instead of an arbitrary identity/noise matrix.
    """
    K = 133
    edges = []
    # COCO body (0-16): nose,l_eye,r_eye,l_ear,r_ear,l_sh,r_sh,l_el,r_el,l_wr,r_wr,
    # l_hip,r_hip,l_knee,r_knee,l_ank,r_ank
    edges += [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (5, 7), (6, 8),
              (7, 9), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14),
              (13, 15), (14, 16)]
    # feet (17-22): l_big_toe, l_small_toe, l_heel, r_big_toe, r_small_toe, r_heel
    edges += [(15, 17), (15, 18), (15, 19), (17, 18),
              (16, 20), (16, 21), (16, 22), (20, 21)]
    # face (23-90, 68 pts, dlib ordering): jaw 0-16, r_brow 17-21, l_brow 22-26,
    # nose 27-35, r_eye 36-41, l_eye 42-47, mouth 48-67
    f = 23

    def chain(pts, closed=False):
        e = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        if closed:
            e.append((pts[-1], pts[0]))
        return e

    jaw = [f + i for i in range(0, 17)]
    r_brow = [f + i for i in range(17, 22)]
    l_brow = [f + i for i in range(22, 27)]
    nose_bridge = [f + i for i in range(27, 31)]
    nose_base = [f + i for i in range(31, 36)]
    r_eye = [f + i for i in range(36, 42)]
    l_eye = [f + i for i in range(42, 48)]
    mouth_outer = [f + i for i in range(48, 60)]
    mouth_inner = [f + i for i in range(60, 68)]
    edges += chain(jaw) + chain(r_brow) + chain(l_brow)
    edges += chain(nose_bridge) + chain(nose_base)
    edges += chain(r_eye, closed=True) + chain(l_eye, closed=True)
    edges += chain(mouth_outer, closed=True) + chain(mouth_inner, closed=True)
    edges += [(0, f + 30), (f + 36, 1), (f + 45, 2)]  # tie face to COCO head points

    # hands (91-111 left, 112-132 right): wrist + 5 fingers x 4 joints each
    def hand_edges(wrist: int):
        e = []
        for finger in range(5):
            start = wrist + 1 + finger * 4
            pts = [wrist] + [start + j for j in range(4)]
            e += [(pts[i], pts[i + 1]) for i in range(4)]
        return e

    edges += hand_edges(91) + hand_edges(112)
    edges += [(9, 91), (10, 112)]  # COCO wrists to hand wrists

    A = torch.eye(K)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


class PoseGraph(nn.Module):
    """Learned-adjacency graph front-end over the pose keypoints (CTR-GCN-style).

    The flat feature vector treats the K keypoints as an unordered bag; this mixes
    information *across* keypoints with a learned soft adjacency A (K x K) before
    the encoder ever sees the sequence, so the model can use skeletal structure
    (hand joints, arm chains) instead of a flat vector. Dimension-preserving and
    residual: output shape == input shape, so `input_proj` is unchanged and the
    block is a no-op at init-ish. Gated by cfg.pose_graph.

    A is initialised from the real COCO-WholeBody-133 skeleton (falls back to
    identity for other keypoint counts) and is excluded from PoseSLT.reset_parameters'
    blanket xavier re-init — see that method's docstring for why this matters.
    """

    def __init__(self, num_kpts: int, chan: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.K = num_kpts
        self.C = chan
        base_adj = _coco_wholebody_133_adjacency() if num_kpts == 133 else torch.eye(num_kpts)
        # one learned adjacency + per-node channel mixer + norm per layer
        self.A = nn.ParameterList(
            [nn.Parameter(base_adj.clone() + 0.01 * torch.randn(num_kpts, num_kpts))
             for _ in range(layers)]
        )
        self.W = nn.ModuleList([nn.Linear(chan, chan) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(chan) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, K*C)
        B, T, F = x.shape
        h = x.view(B, T, self.K, self.C)
        for A, W, ln in zip(self.A, self.W, self.norm):
            An = torch.softmax(A, dim=-1)                       # (K, K) row-normalised
            mixed = torch.einsum("kj,btjc->btkc", An, h)        # neighbour aggregation
            h = ln(h + self.drop(torch.relu(W(mixed))))         # residual + channel mix
        return h.reshape(B, T, F)


class PoseSLT(nn.Module):
    def __init__(self, feat_dim: int, vocab_size: int, cfg):
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        # Optional learned-graph keypoint front-end (dimension-preserving, gated).
        self.pose_graph = None
        if bool(getattr(cfg, "pose_graph", False)):
            nk = int(getattr(cfg, "num_keypoints", 0)) or 0
            if nk > 0 and feat_dim % nk == 0:
                self.pose_graph = PoseGraph(
                    nk, feat_dim // nk,
                    int(getattr(cfg, "pose_graph_layers", 2)), cfg.dropout,
                )
        # Wave 13 part-wise front end (off by default): replaces the flat per-frame vector by 4 x 256 pooled part features.
        self.w13 = None
        if getattr(cfg, "w13_frontend", ""):
            from .w13_frontend import W13Front
            self.w13 = W13Front("skeleton" if cfg.w13_frontend == "sg" else "identity", cfg.dropout)
            feat_dim = self.w13.out_dim
        visual_d = int(getattr(cfg, "visual_feature_dim", 0))
        if visual_d > 0:
            self.visual_adapter = nn.Sequential(
                nn.Linear(feat_dim, visual_d),
                nn.LayerNorm(visual_d),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            )
            transformer_input_dim = visual_d
        else:
            self.visual_adapter = nn.Identity()
            transformer_input_dim = feat_dim
        self.input_proj = nn.Sequential(
            nn.Linear(transformer_input_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.pos_enc = PositionalEncoding(d, dropout=cfg.dropout)
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.transformer = nn.Transformer(
            d_model=d,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        # Disable the nested-tensor fast path. With src_key_padding_mask PyTorch
        # otherwise converts the batch via torch._nested_tensor_from_mask, which
        # emits a prototype-API UserWarning and can subtly differ from the plain
        # padded path. Correctness/reproducibility over a small speedup.
        # NOTE: TransformerEncoder gates on `use_nested_tensor` at forward time;
        # setting `enable_nested_tensor` alone is NOT enough — both must be off.
        for _attr in ("enable_nested_tensor", "use_nested_tensor"):
            if hasattr(self.transformer.encoder, _attr):
                setattr(self.transformer.encoder, _attr, False)
        self.generator = nn.Linear(d, vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        # NOTE: pose_graph.A is a (K, K) matrix (dim=2), so the blanket xavier
        # pass below used to catch it too and overwrite its anatomical/near-identity
        # init with random noise -- the graph front-end then started as a
        # destructive random keypoint-mixer instead of a residual no-op (2026-09-04
        # job 1171052 regression: layers=2/3 both ~1.3-1.7 BLEU below control).
        # Exclude it explicitly so its skeleton-aware init survives.
        graph_A_ids = {id(p) for p in self.pose_graph.A} if self.pose_graph is not None else set()
        for p in self.parameters():
            if p.dim() > 1 and id(p) not in graph_A_ids:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        if self.w13 is not None:
            src = self.w13(src, src_pad)
        if self.pose_graph is not None:
            src = self.pose_graph(src)
        visual = self.visual_adapter(src)
        x = self.pos_enc(self.input_proj(visual) * math.sqrt(self.d_model))
        return self.transformer.encoder(x, src_key_padding_mask=src_pad)

    def decode_step(self, tgt_in, memory, src_pad, tgt_pad):
        L = tgt_in.size(1)
        causal = torch.triu(torch.ones(L, L, device=tgt_in.device, dtype=torch.bool), 1)
        y = self.pos_enc(self.tok_emb(tgt_in) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            y, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.generator(out)

    def forward(self, src, src_pad, tgt_in, tgt_pad, return_memory: bool = False):
        memory = self.encode(src, src_pad)
        logits = self.decode_step(tgt_in, memory, src_pad, tgt_pad)
        # return_memory lets the trainer add a SignCL contrastive loss on the encoder
        # features while keeping DDP's forward() as the single entry point.
        return (logits, memory) if return_memory else logits

    @torch.no_grad()
    def greedy_decode(self, src, src_pad, max_len: int = 64) -> torch.Tensor:
        self.eval()
        B = src.size(0)
        memory = self.encode(src, src_pad)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.decode_step(ys, memory, src_pad, None)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            nxt[done] = PAD
            ys = torch.cat([ys, nxt], dim=1)
            done |= nxt.squeeze(1).eq(EOS)
            if bool(done.all()):
                break
        return ys

    @torch.no_grad()
    def beam_decode(self, src, src_pad, max_len: int = 64, beam: int = 5,
                    length_penalty: float = 0.6):
        """Per-sample beam search. Returns a list (len B) of token-id lists.

        Slower than greedy (used for final eval, not per-epoch): encodes each clip
        once, expands to `beam` hypotheses, and keeps the top-`beam` continuations
        each step. Finished (EOS) hypotheses are scored with GNMT-style length
        normalisation (score / len**length_penalty) so beam doesn't over-favour
        short outputs.
        """
        self.eval()
        device = src.device
        out = []
        for b in range(src.size(0)):
            mem = self.encode(src[b:b + 1], src_pad[b:b + 1])          # (1,T,d)
            mem = mem.expand(beam, -1, -1).contiguous()                # (beam,T,d)
            spad = src_pad[b:b + 1].expand(beam, -1).contiguous()      # (beam,T)
            ys = torch.full((beam, 1), BOS, dtype=torch.long, device=device)
            scores = torch.full((beam,), float("-inf"), device=device)
            scores[0] = 0.0
            finished = []
            for _ in range(max_len - 1):
                logits = self.decode_step(ys, mem, spad, None)         # (beam,L,V)
                logp = torch.log_softmax(logits[:, -1], dim=-1)        # (beam,V)
                V = logp.size(-1)
                cand = (scores.unsqueeze(1) + logp).view(-1)           # (beam*V,)
                top_scores, top_idx = cand.topk(beam)
                beam_idx = torch.div(top_idx, V, rounding_mode="floor")
                tok_idx = top_idx % V
                ys = torch.cat([ys[beam_idx], tok_idx.unsqueeze(1)], dim=1)
                scores = top_scores.clone()
                for i in range(beam):
                    if tok_idx[i].item() == EOS:
                        seq = ys[i].tolist()
                        lp = max(1, len(seq)) ** length_penalty
                        finished.append((scores[i].item() / lp, seq))
                        scores[i] = float("-inf")
                if len(finished) >= beam:
                    break
            if not finished:
                best = int(scores.argmax().item())
                finished.append((0.0, ys[best].tolist()))
            finished.sort(key=lambda x: x[0], reverse=True)
            out.append(finished[0][1])
        return out
