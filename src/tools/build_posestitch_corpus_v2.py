"""PoseStitch v2 Stage B — stitch MM-WLAuslan isolated signs into synthetic continuous clips.

Differences from v1 (each one is a measured deficiency of the v1 run, which scored a neutral 9.66):
  * uses ALL 45,010 DWPose clips (train+valid), i.e. 14 exemplars/gloss instead of 2
  * phrase-aware gloss matching, recovering the 612 multi-word glosses v1 could never match
  * default coverage threshold 0.9 (v1 used 0.6, so up to 40% of content words had no sign
    in the pose yet still appeared in the target text -- supervision to hallucinate)
  * transition budget between signs sampled from a distribution and, per the 2026 stitching
    ablation, deliberately NOT L2-smoothed (smoothing cost that paper 0.9-1.1 BLEU); optional
    boundary trim approximates co-articulation by cutting isolated-clip rest frames
  * consumes LLM-generated NOVEL sentences (Stage A) rather than re-realising real ones

Writes DWPose-format npz (KPS + BBOX) + a manifest, so the existing pipeline consumes it unchanged.
Note: MM-WLAuslan clip ids carry no signer label, so signer-consistent stitching is not possible
here; downstream coord_norm=frame standardises each frame independently, which absorbs most
inter-signer scale/position variation.
"""
from __future__ import annotations
import argparse, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.stitch_lexicon import build_lexicon, coverage, content_words


# ── temporal fidelity (2026-09-05 audit) ────────────────────────────────────
# BUG: Auslan-Daily is 25 fps, MM-WLAuslan is 30 fps, and nothing here ever resampled -- so every
# synthetic clip was dilated 1.2x from the frame-rate mismatch alone, before any linguistics.
# On top of that, continuous signs are ~72% SHORTER than their own citation forms: a paired
# per-gloss comparison over 45 glosses gives citation 80.05 fr @30fps (2.67 s) vs continuous
# 22.49 fr @25fps (0.90 s) = 3.60x in seconds, 4.32x in the raw frames the model actually sees.
# Measured directly: one citation clip averages 79.4 frames while a WHOLE real sentence averages
# 91.2, and our synthetic clips run 250.8.
# Real continuous per-sign duration, from Auslan-Daily's own Sign_Spotting.xlsx (1,069 instances):
# mean 22.49, median 20, p10 10, p90 37 frames @25fps.
_DUR_Q = [(0.00, 8.0), (0.10, 10.0), (0.50, 20.0), (0.90, 37.0), (1.00, 60.0)]


def sample_sign_duration(rng) -> int:
    """Draw a target sign length from the measured continuous distribution via a piecewise-linear
    inverse CDF through its quantiles. Sampling a DISTRIBUTION (rather than applying one constant
    factor) is what SignaVox's duration-predictor stage does, and its ablation shows duration
    planning matters more than the diffusion boundary model."""
    u = float(rng.random())
    for (q0, v0), (q1, v1) in zip(_DUR_Q, _DUR_Q[1:]):
        if u <= q1:
            return max(4, int(round(v0 + (u - q0) / (q1 - q0) * (v1 - v0))))
    return int(_DUR_Q[-1][1])


def resample_clip(k, b, n_out: int):
    """Linear resample along TIME to n_out frames. This is temporal rate correction, NOT the
    boundary L2-smoothing that cost the 2026 stitching paper 0.9-1.1 BLEU: it rescales when each
    sign happens without averaging one sign into the next."""
    T = k.shape[0]
    if T < 2 or n_out < 1 or n_out == T:
        return k, b
    idx = np.linspace(0.0, T - 1.0, n_out)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (idx - lo).astype(np.float32)
    k2 = (1 - w)[:, None, None] * k[lo] + w[:, None, None] * k[hi]
    b2 = (1 - w)[:, None] * b[lo] + w[:, None] * b[hi]
    return k2.astype(np.float32), b2.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="MM-WLAuslan label JSONs")
    ap.add_argument("--npz_dirs", nargs="+", required=True, help="matching DWPose npz dirs")
    ap.add_argument("--sentences", required=True, help="Stage A novel sentences, one per line")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_cov", type=float, default=0.9)
    ap.add_argument("--min_signs", type=int, default=3)
    ap.add_argument("--realizations", type=int, default=1)
    ap.add_argument("--max_clips", type=int, default=0, help="cap output (0 = no cap)")
    ap.add_argument("--trim_frac", type=float, default=0.15,
                    help="fraction of frames trimmed from each end of an isolated clip (rest pose)")
    ap.add_argument("--max_transition", type=int, default=4,
                    help="max interpolated transition frames between consecutive signs (0 = hard cut)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_paren_alias", action="store_true",
                    help="do NOT index a gloss's parenthetical as a standalone alias -- it is "
                         "usually a sense disambiguator, so aliasing stitches wrong-sense signs")
    ap.add_argument("--morphology", action="store_true",
                    help="let inflected words match citation-form glosses (must match Stage A)")
    a = ap.parse_args()
    if len(a.labels) != len(a.npz_dirs):
        raise SystemExit("--labels and --npz_dirs must be parallel lists")
    random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    clips_out = os.path.join(a.out_dir, "clips")
    os.makedirs(clips_out, exist_ok=True)

    lex = build_lexicon(a.labels, a.npz_dirs, morphology=a.morphology,
                        paren_alias=not a.no_paren_alias)
    n_ex = [len(v) for v in lex.gloss2clips.values()]
    print(f"[stitch2] lexicon: {len(lex.gloss2clips)} glosses, {sum(n_ex)} clips, "
          f"median {int(np.median(n_ex))} exemplars/gloss, {len(lex.phrase2gloss)} surface forms",
          flush=True)

    cache = {}

    def load(entry):
        if entry not in cache:
            d, cid = entry
            z = np.load(os.path.join(d, cid + ".npz"))
            k, b = z["KPS"].astype(np.float32), z["BBOX"].astype(np.float32)
            if a.trim_frac > 0 and k.shape[0] > 8:          # drop rest pose at both ends
                cut = int(k.shape[0] * a.trim_frac)
                k, b = k[cut:k.shape[0] - cut], b[cut:b.shape[0] - cut]
            if len(cache) < 20000:                           # bounded LRU-ish cache
                cache[entry] = (k, b)
            return k, b
        return cache[entry]

    rows, n_written, skipped = [], 0, 0
    sentences = [ln.strip() for ln in open(a.sentences, encoding="utf-8") if ln.strip()]
    print(f"[stitch2] {len(sentences)} candidate sentences", flush=True)

    for text in sentences:
        cov, spans = coverage(text, lex)
        if len(spans) < a.min_signs or cov < a.min_cov or not content_words(text):
            skipped += 1
            continue
        for _ in range(a.realizations):
            kps_parts, bbox_parts = [], []
            for si, (_, _, gloss) in enumerate(spans):
                k, b = load(random.choice(lex.gloss2clips[gloss]))
                if si > 0 and a.max_transition > 0 and kps_parts:
                    # transition BUDGET (sampled), not a smoothness objective: linear blend of the
                    # two boundary poses. The 2026 ablation found optimising boundaries for L2
                    # smoothness HURT, so this stays short and randomised.
                    n_t = int(rng.integers(0, a.max_transition + 1))
                    if n_t:
                        prev, nxt = kps_parts[-1][-1], k[0]
                        w = np.linspace(0, 1, n_t + 2)[1:-1][:, None, None]
                        kps_parts.append((1 - w) * prev[None] + w * nxt[None])
                        bbox_parts.append(np.repeat(bbox_parts[-1][-1][None], n_t, axis=0))
                kps_parts.append(k)
                bbox_parts.append(b)
            KPS = np.concatenate(kps_parts, 0).astype(np.float32)
            BBOX = np.concatenate(bbox_parts, 0).astype(np.float32)
            name = f"stitch2_{n_written:06d}.npz"
            np.savez_compressed(os.path.join(clips_out, name), KPS=KPS, BBOX=BBOX)
            rows.append(f"{os.path.join(clips_out, name)}\t{text}")
            n_written += 1
            if a.max_clips and n_written >= a.max_clips:
                break
        if a.max_clips and n_written >= a.max_clips:
            break

    man = os.path.join(a.out_dir, "posestitch_v2_train.tsv")
    open(man, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print(f"[stitch2] wrote {n_written} synthetic clips ({skipped} sentences skipped below "
          f"cov>={a.min_cov}) -> {man}", flush=True)
    if rows:
        ex = np.load(rows[0].split("\t")[0])
        lens = [np.load(r.split("\t")[0])["KPS"].shape[0] for r in rows[:200]]
        print(f"[stitch2] example KPS{ex['KPS'].shape} BBOX{ex['BBOX'].shape} | "
              f"frames over first 200: mean {np.mean(lens):.0f} min {min(lens)} max {max(lens)}",
              flush=True)


if __name__ == "__main__":
    main()
