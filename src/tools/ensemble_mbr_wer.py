"""Decode-time WER reduction for the ensemble WITHOUT hurting BLEU-4.

Goal: lower WER while maintaining/improving BLEU-4 on the S6 ensemble, using only
decode-time methods (no retraining). Two levers, from ONE beam decode:

  1. Length-penalty (LP) sweep — the S6 error profile has Deletions > Insertions
     (mild brevity); the WER-optimal LP can differ from the BLEU-optimal LP.
  2. MBR (Minimum Bayes Risk) decoding — instead of the single highest-probability
     hypothesis, pick the candidate that minimises expected WER against the whole
     n-best (consensus decoding). This directly targets WER.

We decode an n-best ONCE (expensive), then rescore cheaply for every LP and for MBR.
Selection MUST be on validation; the official test is scored only once for the chosen
config. Reuses ensemble_decode's Member (same heterogeneous pose+fusion handling).

  python tools/ensemble_mbr_wer.py --ckpts a.pt,b.pt,... --rgb_feat_dir <dir> \
      --eval_manifest val.tsv --num_keypoints 133 --keypoint_subset all --coord_norm frame \
      --beam 10 --device cpu
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from slt.cli import strict_bool
from slt.preproc_guard import assert_preproc_matches_checkpoints
from slt.data import Vocab, BOS, EOS
from slt.metrics import corpus_bleu4, corpus_wer, detokenize
from slt.engine import resolve_device
from ensemble_decode import Member                    # reuse identical member handling
import sacrebleu, jiwer


def power_mean_logp(lps, w, power):
    """Log of the weighted POWER mean of the members' probabilities: (sum_i w_i q_i**p / sum_i w_i)**(1/p).

    p = 1 is the arithmetic mean (a mixture of the members), p = -1 the harmonic mean; as p -> 0 it tends to the
    geometric mean. The geometric mean itself (p = 0, the rule of every recorded ensemble: the mean of the members'
    log-probabilities) is NOT computed here: with power == 0 ensemble_nbest keeps its original arithmetic.
    Like that default, the result is not renormalised over the vocabulary (it sums to 1 only for p = 1).
    Wave 12d, results/PREREG_WAVE12D_POWERMEAN_2026-10-13.md."""
    assert power != 0 and all(wi > 0 for wi in w), "power mean needs p != 0 and positive weights"
    x = torch.stack([power * l + math.log(wi) for wi, l in zip(w, lps)])
    return (torch.logsumexp(x, 0) - math.log(sum(w))) / power


@torch.no_grad()
def ensemble_nbest(members, mems, max_len, beam, dev, weights=None, power=0.0):
    """Like ensemble_beam but returns the FULL n-best per sample as (raw_logprob, seq).

    raw_logprob is the un-length-normalised summed weighted-mean log-prob, so any LP
    can be applied afterwards as raw/(len**LP).
    power (default 0 = off): combine the members with power_mean_logp instead of the mean log-prob."""
    w = weights if weights is not None else [1.0] * len(members)
    wsum = sum(w)
    B = mems[0][0].size(0)
    outs = []
    for b in range(B):
        mem_b = [(m[b:b + 1].expand(beam, -1, -1).contiguous(),
                  p[b:b + 1].expand(beam, -1).contiguous()) for (m, p) in mems]
        ys = torch.full((beam, 1), BOS, dtype=torch.long, device=dev)
        scores = torch.full((beam,), float("-inf"), device=dev); scores[0] = 0.0
        fin = []
        for _ in range(max_len - 1):
            lps = [mem.step_logp(ys, mm, pp) for mem, (mm, pp) in zip(members, mem_b)]
            if power:                                   # default off (wave 12d)
                logp = power_mean_logp(lps, w, power)
            else:                                       # the recorded rule: same operations, same order as before
                lp_sum = None
                for wi, l in zip(w, lps):
                    l = wi * l
                    lp_sum = l if lp_sum is None else lp_sum + l
                logp = lp_sum / wsum
            V = logp.size(-1)
            cand = (scores.unsqueeze(1) + logp).view(-1)
            top, idx = cand.topk(beam)
            bi = torch.div(idx, V, rounding_mode="floor"); ti = idx % V
            ys = torch.cat([ys[bi], ti.unsqueeze(1)], 1); scores = top.clone()
            for i in range(beam):
                if ti[i].item() == EOS:
                    fin.append((scores[i].item(), ys[i].tolist()))
                    scores[i] = float("-inf")
            if len(fin) >= beam:
                break
        if not fin:                                     # nothing hit EOS -> take current beams
            for i in range(beam):
                if scores[i].item() > float("-inf"):
                    fin.append((scores[i].item(), ys[i].tolist()))
        outs.append(fin)
    return outs


def word_wer(ref_words, hyp_words):
    """Word-level Levenshtein / max(1,len(ref)) — fast pairwise WER for MBR."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref_words[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp_words[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def corpus_scores(hyps_txt, refs_txt):
    h = [detokenize(x) for x in hyps_txt]; r = [detokenize(x) for x in refs_txt]
    bleu = corpus_bleu4(h, r, tokenize="13a")
    wer = corpus_wer(h, r)
    d = jiwer.process_words(r, h)
    return bleu, wer, d.substitutions, d.deletions, d.insertions, len(set(hyps_txt))



class RemappedMember:
    """Wrap a Member whose vocabulary differs from the ensemble's reference vocabulary.

    WHY THIS EXISTS (found 2026-09-09, job 1191189). `main()` used to do
    `vocab = Vocab(cks[0]["vocab"])` and hand that ONE vocab to every member. Models trained on
    different manifests get different vocabularies: the duration-stitch models were trained on the
    joint real+synthetic manifest, so `Vocab.build` sorted by different word frequencies. Both
    vocabularies happen to be size 3063 -- so nothing crashed and nothing warned -- but index 5 is
    'i' for the S6 models and 'the' for the duration models. Averaging logits across them silently
    mixes up which word each index denotes.

    The damage scaled with the number of mismatched members, exactly as observed on validation:
    S6-only 11.707 and dur-only 12.280 (both internally consistent) against S6+1dur 10.496,
    3dur+2rgb 5.482, S6+3dur 5.102, S6+3dur+vlp 2.803. Those four were harness artefacts, and the
    plausible-sounding conclusion they invited -- "mixing model families hurts ensembles" -- would
    have been wrong.

    Remapping runs in BOTH directions: decoder inputs are reference-vocab ids and must be converted
    to the member's own ids before its forward pass, and its output distribution must be permuted
    back into reference-vocab order. Words the member cannot produce get -inf so they contribute
    nothing to the average rather than aliasing onto an unrelated word.
    """

    def __init__(self, member, member_itos, ref_itos, device):
        self.m = member
        self.fusion = getattr(member, "fusion", False)
        m_index = {w: i for i, w in enumerate(member_itos)}
        # ref id -> member id (for decoder inputs); UNK where the member lacks the word
        unk = m_index.get("<unk>", 3)
        self.ref2mem = torch.tensor([m_index.get(w, unk) for w in ref_itos],
                                    dtype=torch.long, device=device)
        # for each REF position, which member index supplies it (-1 = member cannot produce it)
        self.out_src = torch.tensor([m_index.get(w, -1) for w in ref_itos],
                                    dtype=torch.long, device=device)
        self.missing = self.out_src < 0
        self.out_src = self.out_src.clamp(min=0)

    def __getattr__(self, name):
        # Transparent proxy for everything except step_logp. Written as delegation rather than an
        # explicit method list because the first version forwarded only `encode` and crashed on
        # `Member.memory` (job 1195355, 54s). Delegation cannot miss a method.
        return getattr(object.__getattribute__(self, "m"), name)

    def step_logp(self, ys, mem, pad):
        lp = self.m.step_logp(self.ref2mem[ys], mem, pad)
        out = lp.index_select(-1, self.out_src)
        if self.missing.any():
            out = out.masked_fill(self.missing, float("-inf"))
        return out


def _align_vocabs(cks, members, ref_itos, dev, remap):
    """Verify every member shares the reference vocabulary; optionally remap those that do not."""
    out, mismatched = [], []
    for i, (c, m) in enumerate(zip(cks, members)):
        itos = list(c["vocab"])
        if itos == ref_itos:
            out.append(m)
        else:
            mismatched.append(i)
            out.append(RemappedMember(m, itos, ref_itos, dev) if remap else m)
    if mismatched and not remap:
        raise SystemExit(
            f"FATAL: members {mismatched} have a DIFFERENT vocabulary from member 0 "
            f"(sizes {[len(c['vocab']) for c in cks]}). Averaging their logits is meaningless -- "
            f"identical size does NOT imply identical index order. Pass --remap_vocab to align "
            f"them, or ensemble only models trained on the same manifest.")
    if mismatched:
        print(f"[ens] remapped {len(mismatched)} member(s) into member 0's vocabulary: {mismatched}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    cfg0 = Config()
    for name, val in cfg0.to_dict().items():
        if isinstance(val, bool): ap.add_argument(f"--{name}", type=strict_bool, default=val)
        elif val is None: ap.add_argument(f"--{name}", type=str, default=None)
        else: ap.add_argument(f"--{name}", type=type(val), default=val)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--remap_vocab", action="store_true",
                    help="align members whose vocabulary differs from member 0 (see RemappedMember). "
                         "Without it, a mismatch is a hard error rather than silent corruption.")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval_manifest", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--power_mean", type=float, default=0.0,
                    help="0 (default) = the recorded rule, mean of the members' log-probabilities (geometric mean); "
                         "p != 0 = power mean of their probabilities (1 arithmetic, -1 harmonic). Wave 12d.")
    ap.add_argument("--lps", default="0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    ap.add_argument("--mbr_lp", type=float, default=1.0, help="LP used to weight candidates in MBR")
    ap.add_argument("--mbr_temp", type=float, default=1.0)
    ap.add_argument("--dump_dir", default="", help="write per-sample hypotheses (and the full "
                    "n-best) as jsonl so the ENSEMBLE operating point can be error-analysed -- "
                    "without this the headline 11.80/0.907 config leaves no decodes behind, which "
                    "is exactly why the 2026-09-05 WER analysis had to use single-model proxies")
    ap.add_argument("--allow_preproc_mismatch", action="store_true",
                    help="downgrade the preprocessing-agreement check to a warning (deliberate probes only)")
    a = ap.parse_args()
    # Keep decoder-only options out of Config.  This used to rely on a hand-maintained
    # exclusion list, so adding --dump_dir accidentally made the validation-only
    # ensemble sweep exit before decoding.  Config's declared fields are the
    # authoritative allow-list.
    cfg = Config(**{name: getattr(a, name) for name in cfg0.to_dict()})
    if a.eval_manifest: cfg.test_manifest = a.eval_manifest
    dev = resolve_device(cfg.device)
    paths = [p.strip() for p in a.ckpts.split(",") if p.strip()]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    vocab = Vocab(cks[0]["vocab"])
    # each member is built at ITS OWN vocabulary size; _align_vocabs then maps it into member 0's vocabulary
    # (models trained on different manifests differ in size, not only order: e.g. vd real-only 2805 vs joint 2867)
    assert_preproc_matches_checkpoints(cfg, cks, a.allow_preproc_mismatch)
    members = [Member(c, cfg, Vocab(c["vocab"]), dev) for c in cks]
    members = _align_vocabs(cks, members, list(cks[0]["vocab"]), dev, a.remap_vocab)
    any_fusion = any(m.fusion for m in members)
    weights = [float(x) for x in a.weights.split(",")] if a.weights else None
    print(f"[mbr] {len(members)} members fusion={[m.fusion for m in members]} "
          f"beam={a.beam} manifest={os.path.basename(cfg.test_manifest)} dev={dev}", flush=True)
    if a.power_mean:
        print(f"[mbr] POWER MEAN p={a.power_mean:g} instead of the geometric mean (wave 12d)", flush=True)

    if any_fusion:
        # A fusion member with no RGB directory is NOT an error inside RGBDataset: _load_rgb
        # substitutes a single zero frame so attention memory is never empty. That is the right
        # behaviour for one missing clip and a trap for a missing DIRECTORY -- the member silently
        # degrades to pose-only and the run still prints a plausible number. cfg here is built from
        # ARGPARSE DEFAULTS, not from the checkpoints, so `rgb_feat_dir` is "" unless passed, and
        # five recent duration-ensemble sbatch scripts never passed it. Fail loudly instead.
        if not cfg.rgb_feat_dir or not os.path.isdir(cfg.rgb_feat_dir):
            raise SystemExit(
                f"[mbr] FATAL: {sum(m.fusion for m in members)} fusion member(s) present but "
                f"--rgb_feat_dir is {cfg.rgb_feat_dir!r}. Without it every fusion member is fed a "
                f"single ZERO frame and silently degrades to pose-only. Pass --rgb_feat_dir <dir> "
                f"(and --rgb_feature_dim, default 1152).")
        from slt.rgb_fusion import RGBDataset, collate_rgb
        ds = RGBDataset(cfg.test_manifest, vocab, cfg, ""); collate_fn = collate_rgb
        import glob as _glob
        _n = len(_glob.glob(os.path.join(cfg.rgb_feat_dir, "rgb_*.npz")))
        print(f"[mbr] rgb_feat_dir={cfg.rgb_feat_dir} ({_n} feature files)", flush=True)
    else:
        from slt.data import AuslanDailyDataset, collate
        ds = AuslanDailyDataset(cfg.test_manifest, vocab, cfg, ""); collate_fn = collate
    if a.limit: ds.samples = ds.samples[:a.limit]
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens

    # ---- decode n-best ONCE ----
    all_nbest, refs = [], []
    for batch in dl:
        if any_fusion:
            src, src_pad, rgb, rgb_pad, _t, texts = batch
            rgb, rgb_pad = rgb.to(dev), rgb_pad.to(dev)
        else:
            src, src_pad, _t, texts = batch; rgb = rgb_pad = None
        src, src_pad = src.to(dev), src_pad.to(dev)
        mems = [m.memory(src, src_pad, rgb, rgb_pad) for m in members]
        nb = ensemble_nbest(members, mems, mx, a.beam, dev, weights, a.power_mean)
        for cand_list, ref in zip(nb, texts):
            cands = [(raw, vocab.decode(seq)) for raw, seq in cand_list]
            all_nbest.append(cands); refs.append(ref)
    print(f"[mbr] decoded n-best for {len(all_nbest)} samples "
          f"(avg cands={sum(len(c) for c in all_nbest)/max(1,len(all_nbest)):.1f})", flush=True)

    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        with open(os.path.join(a.dump_dir, "nbest.jsonl"), "w", encoding="utf-8") as fh:
            for i, (cands, ref) in enumerate(zip(all_nbest, refs)):
                fh.write(json.dumps({"idx": i, "ref": ref,
                                     "nbest": [[float(sc), h] for sc, h in cands]}) + "\n")
        print(f"[mbr] wrote n-best -> {a.dump_dir}/nbest.jsonl", flush=True)

    def report(tag, hyps):
        bleu, wer, S, D, I, uniq = corpus_scores(hyps, refs)
        print(f"  {tag:16s} BLEU-4={bleu:6.3f}  WER={wer:.4f}  S={S} D={D} I={I}  uniq={uniq}", flush=True)
        if a.dump_dir:
            safe = re.sub(r"[^A-Za-z0-9._=-]", "_", tag)
            with open(os.path.join(a.dump_dir, f"hyps_{safe}.jsonl"), "w", encoding="utf-8") as fh:
                for i, (h, r) in enumerate(zip(hyps, refs)):
                    fh.write(json.dumps({"idx": i, "ref": r, "hyp_beam": h}) + "\n")

    # ---- 1) LP sweep (top-1 by length-normalised score) ----
    print("[mbr] === length-penalty sweep (argmax) ===", flush=True)
    lps = [float(x) for x in a.lps.split(",")]
    for lp in lps:
        hyps = []
        for cands in all_nbest:
            best = max(cands, key=lambda c: c[0] / (max(1, len(c[1].split())) ** lp))
            hyps.append(best[1])
        report(f"LP={lp}", hyps)

    # ---- 2) MBR decoding: minimise expected WER against the weighted n-best ----
    print("[mbr] === MBR (minimise expected WER) ===", flush=True)
    hyps_mbr = []
    for cands in all_nbest:
        texts = [c[1] for c in cands]
        words = [t.split() for t in texts]
        # candidate posterior from length-normalised logprob
        ln = torch.tensor([c[0] / (max(1, len(w)) ** a.mbr_lp) for c, w in zip(cands, words)])
        p = F.softmax(ln / a.mbr_temp, dim=0).tolist()
        best_i, best_risk = 0, float("inf")
        for i in range(len(cands)):
            risk = sum(p[j] * word_wer(words[j], words[i]) for j in range(len(cands)))
            if risk < best_risk:
                best_risk, best_i = risk, i
        hyps_mbr.append(texts[best_i])
    report("MBR-WER", hyps_mbr)

    # ---- 3) MBR-chrF: maximise expected chrF vs the weighted n-best (a BLEU-leaning utility) ----
    import sacrebleu
    print("[mbr] === MBR (maximise expected chrF) ===", flush=True)
    hyps_chrf = []
    for cands in all_nbest:
        texts = [c[1] for c in cands]
        words = [t.split() for t in texts]
        dt = [detokenize(t) for t in texts]
        ln = torch.tensor([c[0] / (max(1, len(w)) ** a.mbr_lp) for c, w in zip(cands, words)])
        p = F.softmax(ln / a.mbr_temp, dim=0).tolist()
        best_i, best_u = 0, float("-inf")
        for i in range(len(cands)):
            u = sum(p[j] * sacrebleu.sentence_chrf(dt[i], [dt[j]]).score for j in range(len(cands)))
            if u > best_u:
                best_u, best_i = u, i
        hyps_chrf.append(texts[best_i])
    report("MBR-chrF", hyps_chrf)


if __name__ == "__main__":
    main()
