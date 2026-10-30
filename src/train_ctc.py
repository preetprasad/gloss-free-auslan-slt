"""Pseudo-gloss CTC anti-collapse experiment (standalone; does not touch shared files).

Adds a CTC head on the encoder, supervised by rule-based content-word pseudo-glosses,
to the exact control translator:  L = CE(decoder) + ctc_weight * CTC(encoder).
CTC forces each encoder timestep to align to a gloss, so the decoder cannot ignore
the encoder (the posterior-collapse failure diagnosed in the deep-dive). Reports test
BLEU + distinct-output count vs the from-scratch control (1.76 greedy / 2.87 beam, 53 unique).
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate, tokenize, PAD
from slt.reference_model import build_model
from slt.engine import evaluate, resolve_device, WarmupInverseSqrt, WarmupCosine

STOP = set("a an the is are am was were be been being to of in on at and or but i you he she "
           "it we they this that these those my your his her its our their do does did not have "
           "has had will would can could should may might must shall there here then than so as "
           "if for with from up out about into over after . , ? ! ; :".split())


GLOSS_MAP = {}   # PGG-SLT: normalised-text -> [gloss tokens] (LLM pseudo-glosses), if loaded


def _normkey(t):
    return " ".join(t.lower().split())


def to_gloss(text):
    if GLOSS_MAP:                                    # PGG-SLT LLM pseudo-glosses override the rule
        g = GLOSS_MAP.get(_normkey(text))
        if g:
            return g
    ws = [w for w in tokenize(text, "whitespace") if w.isalpha() and w not in STOP]
    out = []
    for w in ws:
        if not out or out[-1] != w:
            out.append(w)
    return out or ["<g>"]


def build_gloss_vocab(texts):
    toks = set(["<g>"])
    for t in texts:
        toks.update(to_gloss(t))
    return {g: i + 1 for i, g in enumerate(sorted(toks))}   # 0 = CTC blank


# ── wave 12c "CPO": counterfactual-pose preference (off unless --cf_pref_weight > 0) ─────────────────────────────
HAND_DIMS = slice(273, 399)   # both hands in the 399-D feature row (slt/data.py HAND_FEATURE_SLICES)


def seq_logp(logits, tgt_out):
    """(B,) mean log-probability per reference token, teacher forced, no label smoothing.

    Element-wise on purpose: F.cross_entropy on a (B, V, L) input runs the 2-D NLL kernel, which on CUDA is one of the
    operations torch.use_deterministic_algorithms(True) can refuse; a comparison, a product and a sum cannot be."""
    lp = F.log_softmax(logits, -1)
    hit = tgt_out.unsqueeze(-1) == torch.arange(lp.size(-1), device=lp.device)
    keep = tgt_out.ne(PAD)
    return ((lp * hit).sum(-1) * keep).sum(1) / keep.sum(1).clamp_min(1)


def cf_negatives(src, src_pad, texts):
    """Three counterfactual inputs per row, as (src, src_pad, usable) with usable a (B,) bool:
    (a) the pose of the row's neighbour, with the neighbour's mask;
    (b) the row's own pose reversed in time within its valid frames;
    (c) the row's own pose with the neighbour's hand track, resampled to the row's length.
    The neighbour is the next-shorter clip of the batch (the longest clip for the shortest; ties in batch order), so a
    negative differs from the true input in what is signed and as little as possible in how long the clip is.
    (a) and (c) are unusable where the neighbour has the same target sentence (its pose is then a correct input)."""
    B, T, _ = src.shape
    L = (~src_pad).sum(1)
    ar = torch.arange(T, device=src.device)[None]
    order = torch.argsort(L, stable=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(B, device=src.device)
    nb = order[(rank - 1) % B]
    key = [_normkey(t) for t in texts]
    other = torch.tensor([key[i] != key[j] for i, j in enumerate(nb.tolist())], device=src.device)
    nb_src, nb_pad, nb_L = src[nb], src_pad[nb], L[nb]
    rev = (L[:, None] - 1 - ar).clamp_min(0)
    src_rev = torch.where(src_pad[..., None], src, src.gather(1, rev[..., None].expand_as(src)))
    pos = (ar * (nb_L - 1).clamp_min(0)[:, None].float() / (L - 1).clamp_min(1)[:, None].float()).round().long()
    pos = torch.minimum(pos, (nb_L - 1).clamp_min(0)[:, None])
    hands = nb_src[..., HAND_DIMS].gather(1, pos[..., None].expand(B, T, HAND_DIMS.stop - HAND_DIMS.start))
    src_hand = src.clone()
    src_hand[..., HAND_DIMS] = torch.where(src_pad[..., None], src[..., HAND_DIMS], hands)
    return [(nb_src, nb_pad, other), (src_rev, src_pad, L > 1), (src_hand, src_pad, other)]


def cf_preference(model, ref_model, src, src_pad, tgt_in, tgt_out, tgt_pad, logits, texts, beta):
    """-log sigmoid(beta * margin), margin = (lp - ref)(true pose) - mean over usable negatives of (lp - ref)(negative).

    lp is the model's and ref the frozen starting checkpoint's per-token log-probability of the REFERENCE sentence.
    Nothing is sampled, so the term cannot be raised by producing shorter or more generic output."""
    with torch.no_grad():
        ref_true = seq_logp(ref_model(src, src_pad, tgt_in, tgt_pad, return_memory=True)[0], tgt_out)
    d_true = seq_logp(logits, tgt_out) - ref_true
    d_neg, n_neg = torch.zeros_like(d_true), torch.zeros_like(d_true)
    for s, sp, ok in cf_negatives(src, src_pad, texts):
        with torch.no_grad():
            r = seq_logp(ref_model(s, sp, tgt_in, tgt_pad, return_memory=True)[0], tgt_out)
        d = seq_logp(model(s, sp, tgt_in, tgt_pad, return_memory=True)[0], tgt_out) - r
        d_neg = d_neg + torch.where(ok, d, torch.zeros_like(d))
        n_neg = n_neg + ok.float()
    margin = d_true - d_neg / n_neg.clamp_min(1.0)
    use = (n_neg > 0).float()
    n_use = use.sum().clamp_min(1.0)
    return (F.softplus(-beta * margin) * use).sum() / n_use, float((margin.detach() * use).sum() / n_use)


@torch.no_grad()
def val_ce(model, dl, dev):
    """Teacher-forced cross-entropy per reference token over a loader (leaves the model in eval mode)."""
    model.eval()
    tot = n = 0
    for src, src_pad, tgt, _ in dl:
        src, src_pad, tgt = src.to(dev), src_pad.to(dev), tgt.to(dev)
        logits = model(src, src_pad, tgt[:, :-1], tgt[:, :-1].eq(PAD), return_memory=True)[0]
        tot += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                                     ignore_index=PAD, reduction="sum"))
        n += int(tgt[:, 1:].ne(PAD).sum())
    return tot / max(n, 1)


def _strict_bool(x):
    """Accept only true/false: the old `x.lower() == "true"` silently turned "1" or "yes" into False."""
    v = str(x).strip().lower()
    if v not in ("true", "false"):
        raise argparse.ArgumentTypeError(f"expected true or false, got {x!r}")
    return v == "true"


# Config fields that nothing on this trainer's path reads (train_ctc.py, slt/data|model|reference_model|rgb_fusion,
# and the parts of slt/engine it imports) -- found by scanning the source, 2026-09-22. Config is shared with
# train.py / train_mbart.py / train_llm.py / train_vq*.py, so argparse exposes them all here. If you implement one,
# delete it from this tuple.
UNREAD_FIELDS = (
    "pretrained_target_embeddings", "mbart_name", "mbart_lora_r", "mbart_train_lora", "gaslt_similarity_weight",
    "semantic_model", "semantic_cache", "length_bucket_batches", "grad_accum_steps", "plateau_patience",
    "plateau_factor", "lr_min", "loss_normalization", "encoder_lr_scale", "vlp_aux_checkpoint", "vlp_aux_weight",
    "signcl_weight", "signcl_margin", "signcl_zipf", "reference_bleu4", "gate_fraction", "bleu4_gate",
    "freeze_encoder", "llm_name", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "llm_lr",
    "pose_encoder_lr_scale", "max_gen_len") + tuple(n for n in Config().to_dict() if n.startswith("vq_"))


def check_flags(cfg):
    """Refuse flag combinations that would otherwise be silently ignored (research round 2, R2-P0)."""
    default = Config().to_dict()
    ignored = [f"--{n}" for n in UNREAD_FIELDS if getattr(cfg, n) != default[n]]
    if ignored:
        raise SystemExit(f"train_ctc.py never reads {', '.join(ignored)}; use the trainer that implements it")
    for name in ("init_encoder", "resume_checkpoint", "gloss_map", "aug_segment_file"):
        path = getattr(cfg, name, "")
        if path and not os.path.exists(path):
            raise SystemExit(f"--{name} {path} does not exist (it used to be skipped silently)")
    if not 0.0 <= cfg.aug_segment_speed <= 0.5:
        raise SystemExit("--aug_segment_speed must lie in [0, 0.5]")
    if (cfg.aug_segment_speed > 0) != bool(cfg.aug_segment_file):
        raise SystemExit("--aug_segment_speed and --aug_segment_file go together (per-sign tempo needs the sign bounds)")
    if cfg.resume_ctc_head and not cfg.resume_checkpoint:
        raise SystemExit("--resume_ctc_head needs --resume_checkpoint")
    if cfg.cf_pref_weight < 0 or cfg.cf_stop_ce < 0:
        raise SystemExit("--cf_pref_weight and --cf_stop_ce must be >= 0")
    if cfg.cf_pref_weight > 0:
        if cfg.cf_beta <= 0:
            raise SystemExit("--cf_beta must be > 0")
        if not cfg.resume_checkpoint:
            raise SystemExit("--cf_pref_weight needs --resume_checkpoint (its model is the frozen reference)")
        if not (cfg.model_arch == "generic" and cfg.keypoint_subset == "all" and cfg.num_keypoints == 133
                and cfg.use_confidence and not cfg.use_velocity and int(cfg.temporal_pool) == 1):
            raise SystemExit("--cf_pref_weight needs model_arch generic and the 399-D layout (133 keypoints, subset all, "
                             "confidence on, no velocity, temporal_pool 1): the hand negative swaps feature dims 273:399")
    if cfg.w13_input not in ("", "b133", "u69") or cfg.w13_frontend not in ("", "tc", "sg"):
        raise SystemExit("--w13_input must be '', 'b133' or 'u69' and --w13_frontend '', 'tc' or 'sg'")
    if cfg.w13_frontend and cfg.w13_input != "u69":
        raise SystemExit("--w13_frontend needs --w13_input u69 (the front end reads the 69-point layout)")
    if cfg.w13_input:
        want = dict(model_arch="generic", num_keypoints=133, keypoint_subset="all", input_representation="pose",
                    use_confidence=True, use_velocity=False, temporal_pool=1, coord_norm="frame", conf_threshold=0.0,
                    hand_local_norm=False, hierarchical_pose_norm=False, pose_graph=False, visual_feature_dim=0,
                    drop_zero_frames=True, rgb_feat_dir="", aug_hand_drop=0.0, aug_hand_span=0.0, aug_segment_speed=0.0,
                    cf_pref_weight=0.0, init_encoder="", resume_checkpoint="")
        bad = {k: getattr(cfg, k) for k, v in want.items() if getattr(cfg, k) != v}
        if bad:
            raise SystemExit(f"--w13_input is only defined for the adopted 399-D layout; these settings differ: {bad} "
                             "(coord_norm stays 'frame' so the saved cfg names the adopted setting; the transform ignores it)")
    hand_mask = cfg.aug_hand_drop > 0 or cfg.aug_hand_span > 0
    if cfg.eval_every < 1:
        raise SystemExit("--eval_every must be >= 1")
    if hand_mask and not cfg.pose_aug:
        raise SystemExit("--aug_hand_drop/--aug_hand_span need --pose_aug True (hand masking is part of the "
                         "training augmentation)")
    if getattr(cfg, "rgb_feat_dir", ""):
        # slt/rgb_fusion.py builds its own loaders: it never sets pose_augment, calls set_epoch or attaches paraphrases.
        ignored = [name for name, on in (("--pose_aug", cfg.pose_aug), ("--aug_hand_drop/--aug_hand_span", hand_mask),
                                         ("--paraphrase_prob", cfg.paraphrase_prob > 0),
                                         ("--vocab_file", bool(cfg.vocab_file)), ("--save_last", cfg.save_last),
                                         ("--eval_every", cfg.eval_every != 1),
                                         ("--aug_segment_speed", cfg.aug_segment_speed > 0),
                                         ("--resume_ctc_head", cfg.resume_ctc_head),
                                         ("--cf_pref_weight/--cf_stop_ce", cfg.cf_pref_weight > 0 or cfg.cf_stop_ce > 0),
                                         ) if on]
        if ignored:
            raise SystemExit(f"RGB-fusion training (--rgb_feat_dir) never applies {', '.join(ignored)}; "
                             "remove the flag(s) or implement them in slt/rgb_fusion.py")


def parse():
    cfg = Config()
    p = argparse.ArgumentParser()
    for name, val in cfg.to_dict().items():
        if isinstance(val, bool):
            p.add_argument(f"--{name}", type=_strict_bool, default=val)
        elif val is None:
            p.add_argument(f"--{name}", type=str, default=None)
        else:
            p.add_argument(f"--{name}", type=type(val), default=val)
    p.add_argument("--ctc_weight", type=float, default=0.3)
    p.add_argument("--pseudo_token_drop", type=float, default=0.0,
                   help="deterministically drop this fraction of rule-based pseudo-content tokens per training epoch")
    p.add_argument("--limit_samples", type=int, default=0)
    p.add_argument("--swa_last_k", type=int, default=0, help="save last-K epoch ckpts for SWA averaging")
    a = p.parse_args()
    if not 0.0 <= a.pseudo_token_drop < 1.0:
        p.error("--pseudo_token_drop must be in [0, 1)")
    extra = {"ctc_weight", "pseudo_token_drop", "limit_samples", "swa_last_k"}
    cfg = Config(**{k: v for k, v in vars(a).items() if k not in extra})
    return cfg, a


def main():
    cfg, a = parse()
    check_flags(cfg)
    if getattr(cfg, "rgb_feat_dir", ""):          # Stage B: two-stream pose+RGB fusion
        from slt.rgb_fusion import train_rgb_fusion
        return train_rgb_fusion(cfg, a)
    dev = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    if getattr(cfg, "deterministic", False):
        # 2026-09-10ab measured beam-5 gaps up to 1.18 between two runs with everything pinned but cuDNN.
        # CUBLAS_WORKSPACE_CONFIG must be set before the first cuBLAS handle is created, i.e. here.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only MUST be False. With warn_only=True, torch 2.5.1's memory-efficient attention backward only
        # warns and keeps split-K accumulation (attention_backward.cu:652-664), so the A/A run 6593 diverged
        # at epoch 1 (131/132 tensors differ). warn_only=False forces num_splits_key=1 -- deterministic -- and
        # makes any other non-deterministic op RAISE instead of silently passing. CTC is already on CPU (L265).
        torch.use_deterministic_algorithms(True, warn_only=False)
        print(f"[determinism] ON (seed {cfg.seed}, CUBLAS_WORKSPACE_CONFIG="
              f"{os.environ['CUBLAS_WORKSPACE_CONFIG']}) -- reproducible; A/A job 6952 was not slower")

    if getattr(cfg, "gloss_map", "") and os.path.exists(cfg.gloss_map):   # PGG-SLT LLM glosses
        for ln in open(cfg.gloss_map, encoding="utf-8"):
            if "\t" in ln:
                k, v = ln.rstrip("\n").split("\t", 1)
                toks = v.split()
                if toks:
                    GLOSS_MAP[_normkey(k)] = toks
        print(f"[pgg] loaded {len(GLOSS_MAP)} LLM pseudo-glosses from {cfg.gloss_map}", flush=True)

    train_txt = [ln.rstrip("\n").split("\t")[1] for ln in open(cfg.train_manifest) if ln.strip()]
    if cfg.vocab_file:
        # A fixed vocabulary shared by every stage and arm: Vocab.build orders words by frequency IN THE MANIFEST, so a
        # pre-train and a fine-tune on different manifests would index the same word differently.
        vocab = Vocab(json.load(open(cfg.vocab_file)), cfg.text_tokenization, cfg.sentencepiece_model)
        oov = collections.Counter(w for t in train_txt
                                  for w in tokenize(t, cfg.text_tokenization, vocab.sentencepiece) if w not in vocab.stoi)
        if oov:
            raise SystemExit(f"--vocab_file {cfg.vocab_file}: {sum(oov.values())} training tokens are not in it, "
                             f"e.g. {oov.most_common(5)}")
        print(f"[vocab] fixed vocabulary from {cfg.vocab_file}: {len(vocab)} entries, 0 training tokens outside it",
              flush=True)
    else:
        vocab = Vocab.build(train_txt, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
    gloss2idx = build_gloss_vocab(train_txt)
    print(f"[ctc] vocab={len(vocab)} gloss_vocab={len(gloss2idx)} device={dev} feat_dim={cfg.feature_dim()}", flush=True)
    # Frequency-aware weighting: upweight pseudo-gloss targets by inverse corpus frequency so the
    # model attends to rare/long-tail content words (targets the unseen-target ~6 ceiling). Off by default.
    gwt = None
    if cfg.ctc_freq_weight > 0:
        from collections import Counter
        gfreq = Counter(g for t in train_txt for g in to_gloss(t))
        gwt = {g: c ** (-cfg.ctc_freq_weight) for g, c in gfreq.items()}
        print(f"[ctc] freq-aware weighting ON (alpha={cfg.ctc_freq_weight}); "
              f"gloss freq {min(gfreq.values())}..{max(gfreq.values())} over {len(gfreq)} glosses", flush=True)

    # Multi-reference paraphrase augmentation: keep only paraphrases whose tokens are
    # all already in the reference vocabulary, so the output space (and therefore the
    # comparison against the control) is unchanged and the decoder never learns to emit UNK.
    PARA_MAP = {}
    if getattr(cfg, "paraphrase_map", "") and os.path.exists(cfg.paraphrase_map):
        known = set(vocab.itos)
        kept = dropped = 0
        for ln in open(cfg.paraphrase_map, encoding="utf-8"):
            if "\t" not in ln:
                continue
            k, v = ln.rstrip("\n").split("\t", 1)
            v = v.strip()
            if not v or _normkey(v) == _normkey(k):
                continue
            if all(t in known for t in tokenize(v, cfg.text_tokenization)):
                PARA_MAP.setdefault(_normkey(k), []).append(v)
                kept += 1
            else:
                dropped += 1
        print(f"[para] {kept} in-vocab paraphrases over {len(PARA_MAP)} sentences "
              f"({dropped} dropped as out-of-vocab), prob={cfg.paraphrase_prob}", flush=True)

    def loader(manifest, shuffle, limit=0, paraphrase=False, augment=False):
        ds = AuslanDailyDataset(manifest, vocab, cfg, "")
        ds.pose_augment = bool(augment)   # pose augmentation touches the TRAIN loader only
        if limit:
            ds.samples = ds.samples[:limit]
        if paraphrase and PARA_MAP and cfg.paraphrase_prob > 0:
            ds.paraphrases = PARA_MAP
            ds.paraphrase_prob = float(cfg.paraphrase_prob)
        if augment and cfg.aug_segment_speed > 0:   # wave 12c ST: per-sign tempo on the stitched TRAIN rows only
            n_seg = ds.attach_segments(cfg.aug_segment_file)
            print(f"[aug] per-sign tempo ±{cfg.aug_segment_speed} on {n_seg} stitched rows of {len(ds.samples)} "
                  f"({cfg.aug_segment_file})", flush=True)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, collate_fn=collate)

    train_dl = loader(cfg.train_manifest, True, a.limit_samples, paraphrase=True, augment=True)
    if getattr(cfg, "pose_aug", False):
        print(f"[aug] pose augmentation ON (train only): speed ±{cfg.aug_speed}, "
              f"rotation ±{cfg.aug_rotate_deg} deg, x/y scale ±{cfg.aug_scale}, "
              f"hand drop {cfg.aug_hand_drop}, hand span {cfg.aug_hand_span}", flush=True)
    val_dl = loader(cfg.val_manifest, False)
    hand_mask = bool(cfg.pose_aug) and (cfg.aug_hand_drop > 0 or cfg.aug_hand_span > 0)

    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev)
    if cfg.init_encoder and os.path.exists(cfg.init_encoder):   # optional VLP/ISLR pretrained encoder
        ck = torch.load(cfg.init_encoder, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck); ref = model.state_dict()
        pref = ("input_proj", "pos_enc", "transformer.encoder", "pose_adapter",
                "spatial_embedding", "source_position", "encoder_layers", "encoder_norm")
        enc = {k: v for k, v in sd.items() if k.startswith(pref) and k in ref and v.shape == ref[k].shape}
        if not enc:
            raise SystemExit(f"--init_encoder {cfg.init_encoder}: 0 tensors match this model (wrong arch/keypoints?)")
        model.load_state_dict(enc, strict=False)
        print(f"[init] loaded {len(enc)} pretrained encoder tensors from {cfg.init_encoder}", flush=True)
    rk = None
    if getattr(cfg, "resume_checkpoint", "") and os.path.exists(cfg.resume_checkpoint):
        # Full-model warm-start (e.g. PoseStitch pre-train -> fine-tune). Shape-filter so a different
        # pre-training vocab (different decoder embed/output dims) doesn't crash the load: the encoder
        # + vocab-independent decoder body transfer; only the vocab-tied layers re-initialise.
        rk = torch.load(cfg.resume_checkpoint, map_location="cpu", weights_only=False)
        ck_itos = rk.get("vocab") if isinstance(rk, dict) else None
        if ck_itos is not None and list(ck_itos) != list(vocab.itos):
            if len(ck_itos) == len(vocab.itos):
                raise SystemExit(f"--resume_checkpoint {cfg.resume_checkpoint}: same vocabulary size, different order -- "
                                 "the word embeddings and output layer would load scrambled. Use one --vocab_file.")
            print(f"[resume] vocabulary differs ({len(ck_itos)} vs {len(vocab)}): vocab-tied layers re-initialise",
                  flush=True)
        elif ck_itos is not None:
            print(f"[resume] vocabulary identical ({len(vocab)}): vocab-tied layers transfer", flush=True)
        sd = rk.get("model", rk); ref = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in ref and v.shape == ref[k].shape}
        if not keep:
            raise SystemExit(f"--resume_checkpoint {cfg.resume_checkpoint}: 0 tensors match this model")
        model.load_state_dict(keep, strict=False)
        print(f"[resume] full-model warm-start: loaded {len(keep)}/{len(sd)} shape-matched tensors "
              f"from {cfg.resume_checkpoint}", flush=True)
    ctc_head = nn.Linear(cfg.d_model, len(gloss2idx) + 1).to(dev)   # +1 blank(0)
    if cfg.resume_ctc_head:   # a continuation: the head would otherwise restart from random weights
        if not isinstance(rk, dict) or "ctc_head" not in rk or rk.get("gloss2idx") != gloss2idx \
                or list(rk.get("vocab", [])) != list(vocab.itos):
            raise SystemExit(f"--resume_ctc_head: {cfg.resume_checkpoint} has no CTC head for exactly this CTC and "
                             "word vocabulary")
        ctc_head.load_state_dict(rk["ctc_head"])
        print(f"[resume] CTC head restored ({len(gloss2idx)} classes + blank)", flush=True)
    ref_model = None
    if cfg.cf_pref_weight > 0:   # frozen reference = the starting checkpoint
        if len(keep) != len(model.state_dict()):
            raise SystemExit("--cf_pref_weight: the resume checkpoint must supply every model tensor")
        ref_model = copy.deepcopy(model).eval().requires_grad_(False)
        print(f"[cpo] counterfactual-pose preference ON: weight {cfg.cf_pref_weight}, beta {cfg.cf_beta}, frozen "
              f"reference = {cfg.resume_checkpoint}", flush=True)
    params = list(model.parameters()) + list(ctc_head.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * max(1, len(train_dl))
    sched = (WarmupCosine(opt, cfg.lr, cfg.warmup_steps, steps) if cfg.lr_schedule == "cosine"
             else WarmupInverseSqrt(opt, cfg.d_model, cfg.warmup_steps))
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    ctc_none = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")   # for per-sample freq weighting
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    nparams = sum(p.numel() for p in params) / 1e6
    print(f"[ctc] params={nparams:.2f}M train_batches={len(train_dl)} "
          f"pseudo_token_drop={a.pseudo_token_drop:.3f}", flush=True)

    def gloss_targets(texts, epoch):
        ids, lens, wts = [], [], []
        for t in texts:
            toks = to_gloss(t)
            # This is an auditable robustness ablation, not a claim that English-order
            # content words are Auslan glosses.  Stable per (seed, epoch, sentence)
            # sampling makes its pseudo-target corruption reproducible across workers.
            if a.pseudo_token_drop and len(toks) > 1:
                digest = hashlib.blake2b(
                    f"{cfg.seed}:{epoch}:{t}".encode("utf-8"), digest_size=16
                ).digest()
                rng = random.Random(int.from_bytes(digest, "big"))
                kept = [tok for tok in toks if rng.random() >= a.pseudo_token_drop]
                if not kept:
                    kept = [toks[int.from_bytes(digest[:8], "big") % len(toks)]]
                toks = kept
            g = [gloss2idx.get(x, gloss2idx["<g>"]) for x in toks]
            ids.extend(g); lens.append(len(g))
            if gwt is not None:
                wts.append(sum(gwt.get(x, 1.0) for x in toks) / max(1, len(toks)))
        W = torch.tensor(wts, dtype=torch.float32) if gwt is not None else None
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lens, dtype=torch.long), W

    best = {"bleu4": -1, "wer": 9, "epoch": 0}; history = []
    ce0 = None
    if cfg.cf_stop_ce > 0:
        ce0 = val_ce(model, val_dl, dev)
        print(f"[cpo] validation CE before training {ce0:.4f}", flush=True)
    for ep in range(1, cfg.epochs + 1):
        model.train(); ctc_head.train(); tot = tce = tctc = n = 0
        tcf = tmargin = 0.0
        train_dl.dataset.set_epoch(ep)   # re-draws paraphrase targets and pose augmentation each epoch (no-op when off)
        hand_stats = torch.zeros(2, dtype=torch.long, device=dev)   # [hand-frames kept, hand-frames seen]
        for src, src_pad, tgt, texts in train_dl:
            src, src_pad, tgt = src.to(dev), src_pad.to(dev), tgt.to(dev)
            if hand_mask:   # realised keep fraction: a masked hand is an all-zero 63-dim block on a valid frame
                valid = ~src_pad
                zero = src[..., 273:399].reshape(src.size(0), src.size(1), 2, 63).eq(0).all(-1)
                hand_stats += torch.stack([(~zero & valid.unsqueeze(-1)).sum(), 2 * valid.sum()])
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_pad = tgt_in.eq(PAD)
            logits, memory = model(src, src_pad, tgt_in, tgt_pad, return_memory=True)
            loss_ce = ce(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss_rdrop = torch.zeros((), device=dev)
            if cfg.r_drop_alpha > 0:                       # R-Drop: 2nd dropout pass + symmetric KL
                logits2, _ = model(src, src_pad, tgt_in, tgt_pad, return_memory=True)
                loss_ce = 0.5 * (loss_ce + ce(logits2.reshape(-1, logits2.size(-1)), tgt_out.reshape(-1)))
                lp1, lp2 = F.log_softmax(logits, -1), F.log_softmax(logits2, -1)
                kl = 0.5 * (F.kl_div(lp1, lp2.exp(), reduction="none")
                            + F.kl_div(lp2, lp1.exp(), reduction="none")).sum(-1)
                m = (~tgt_out.eq(PAD)).float()
                loss_rdrop = (kl * m).sum() / m.sum().clamp_min(1.0)
            # CTC on CPU (MPS lacks a CTC kernel); grads flow back to MPS params
            logp = torch.log_softmax(ctc_head(memory), dim=-1).transpose(0, 1).float().cpu()
            in_len = (~src_pad).sum(1).cpu()
            tgt_ids, tgt_len, tgt_w = gloss_targets(texts, ep)
            if tgt_w is not None:                          # freq-aware: per-sample weighted mean
                per = ctc_none(logp, tgt_ids, in_len, tgt_len)      # (B,) on CPU
                w = tgt_w / tgt_w.mean().clamp_min(1e-6)            # normalise to mean 1 (keeps scale)
                loss_ctc = (per * w).mean()
            else:
                loss_ctc = ctc(logp, tgt_ids, in_len, tgt_len)
            loss = loss_ce + a.ctc_weight * loss_ctc.to(dev) + cfg.r_drop_alpha * loss_rdrop
            if ref_model is not None:
                loss_cf, margin = cf_preference(model, ref_model, src, src_pad, tgt_in, tgt_out, tgt_pad, logits,
                                                texts, cfg.cf_beta)
                loss = loss + cfg.cf_pref_weight * loss_cf
                tcf += float(loss_cf.detach()) * src.size(0); tmargin += margin * src.size(0)
            sched.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); sched.step()
            b = src.size(0); tot += loss.item() * b; tce += loss_ce.item() * b
            tctc += float(loss_ctc.detach()) * b; n += b
        do_eval = ep % cfg.eval_every == 0 or ep == cfg.epochs
        if do_eval:
            bleu, wer, hyps, _ = evaluate(model, val_dl, vocab, dev, max_len=mx,
                                          bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
            uniq = len(set(hyps))
            print(f"[ctc ep {ep:03d}] loss={tot/n:.3f} ce={tce/n:.3f} ctc={tctc/n:.3f} "
                  f"val_bleu4={bleu:.3f} wer={wer:.3f} uniq={uniq}", flush=True)
            history.append({"epoch": ep, "bleu4": bleu, "wer": wer, "uniq": uniq})
        else:
            print(f"[ctc ep {ep:03d}] loss={tot/n:.3f} ce={tce/n:.3f} ctc={tctc/n:.3f} (no validation this epoch)",
                  flush=True)
            history.append({"epoch": ep, "loss": tot / n})
        if hand_mask:
            keep = hand_stats[0].item() / max(1, hand_stats[1].item())
            history[-1]["hand_keep"] = keep
            print(f"[aug ep {ep:03d}] realised hand keep fraction {keep:.4f}", flush=True)
        if ref_model is not None:
            history[-1]["cf_loss"] = tcf / n; history[-1]["cf_margin"] = tmargin / n
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
            print(f"[cpo ep {ep:03d}] preference loss {tcf / n:.4f} mean margin {tmargin / n:+.4f} nats/token "
                  f"peak GPU {peak:.2f} GiB", flush=True)
        cf_stop = False
        if ce0 is not None and ep == 3:
            ce3 = val_ce(model, val_dl, dev)
            history[-1]["val_ce"] = ce3; history[-1]["val_ce_start"] = ce0
            cf_stop = ref_model is not None and ce3 - ce0 > cfg.cf_stop_ce
            print(f"[cpo ep {ep:03d}] validation CE {ce3:.4f} (start {ce0:.4f}, change {ce3 - ce0:+.4f})"
                  + (" -> STOP: rise above --cf_stop_ce" if cf_stop else ""), flush=True)
        if do_eval and bleu > best["bleu4"]:
            best = {"bleu4": bleu, "wer": wer, "epoch": ep}
            torch.save({"model": model.state_dict(), "ctc_head": ctc_head.state_dict(),
                        "vocab": vocab.itos, "cfg": cfg.to_dict(), "gloss2idx": gloss2idx},
                       os.path.join(cfg.out_dir, "best.pt"))
        if a.swa_last_k and ep > cfg.epochs - a.swa_last_k:    # SWA: save last-K epoch ckpts
            torch.save({"model": model.state_dict(), "vocab": vocab.itos, "cfg": cfg.to_dict()},
                       os.path.join(cfg.out_dir, f"swa_ep{ep}.pt"))
        json.dump({"status": "running", "epoch": ep, "epochs": cfg.epochs,
                   "best": best, "history": history,
                   "pseudo_token_drop": a.pseudo_token_drop,
                   "pseudo_target_order": "english_spoken_order"},
                  open(os.path.join(cfg.out_dir, "progress.json"), "w"), indent=2)
        if cf_stop:
            break

    if cfg.save_last:
        torch.save({"model": model.state_dict(), "ctc_head": ctc_head.state_dict(), "vocab": vocab.itos,
                    "cfg": cfg.to_dict(), "gloss2idx": gloss2idx, "epoch": cfg.epochs},
                   os.path.join(cfg.out_dir, "last.pt"))
        print(f"[save] final epoch {cfg.epochs} model -> last.pt", flush=True)

    # test on best
    ck = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    if cfg.test_manifest:
        test_dl = loader(cfg.test_manifest, False)
        bleu, wer, hyps, _ = evaluate(model, test_dl, vocab, dev, max_len=mx,
                                      bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
        test = {"bleu4": bleu, "wer": wer, "uniq": len(set(hyps)), "from_epoch": best["epoch"]}
        print(f"[ctc TEST] bleu4={bleu:.3f} wer={wer:.3f} distinct_outputs={len(set(hyps))}", flush=True)
        prog = json.load(open(os.path.join(cfg.out_dir, "progress.json")))
        prog["test"] = test
    else:
        prog = json.load(open(os.path.join(cfg.out_dir, "progress.json")))
    prog["status"] = "done"
    prog["test_set_used"] = bool(cfg.test_manifest)
    json.dump(prog, open(os.path.join(cfg.out_dir, "progress.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
