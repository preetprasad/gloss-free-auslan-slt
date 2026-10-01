"""PoseStitch v2 Stage B — stitch MM-WLAuslan isolated signs into synthetic continuous clips.

Differences from v1 (each one is a measured deficiency of the v1 run, which scored a neutral 9.66):
  * uses ALL 45,010 DWPose clips (train+valid), i.e. 14 exemplars/gloss instead of 2
  * phrase-aware gloss matching, recovering the 612 multi-word glosses v1 could never match
  * default coverage threshold 0.9 (v1 used 0.6, so up to 40% of content words had no sign
    in the pose yet still appeared in the target text -- supervision to hallucinate)
  * transition budget between signs sampled from a distribution and, per Robotka et al. (2026,
    arXiv:2606.11925), deliberately NOT L2-smoothed (their SLT ablation found visually smoother
    transitions hurt downstream BLEU); optional
    boundary trim approximates co-articulation by cutting isolated-clip rest frames
  * consumes LLM-generated NOVEL sentences (Stage A) rather than re-realising real ones

Writes DWPose-format npz (KPS + BBOX) + a manifest, so the existing pipeline consumes it unchanged.
Note: MM-WLAuslan clip ids carry no signer label, so signer-consistent stitching is not possible
here; downstream coord_norm=frame standardises each frame independently, which absorbs most
inter-signer scale/position variation.
"""
from __future__ import annotations
import argparse, json, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.stitch_lexicon import build_lexicon, coverage, content_words, words, STOP


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
    boundary L2-smoothing that hurt the 2026 stitching ablation: it rescales when each
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


def motion_resample_clip(k, b, n_out: int):
    """Resample on a CUMULATIVE-MOTION timeline instead of a uniform one (E0, 2026-09-09).

    Uniform `resample_clip` fixes total DURATION but preserves the citation clip's velocity
    PROFILE, so the sign ends up the right length while still being far too static inside.
    E0 measured exactly that on the shipped `--sign_dur` corpus: median frame speed 0.029 vs
    real 0.056, and 42% of frames below the real-corpus pause threshold vs 12% for real
    continuous signing.  Low-motion frames are 54.5% of a citation clip and are spread
    THROUGHOUT it (ends 51.6% vs middle 55.8%), so end/energy trimming cannot remove them.

    Sampling at equal increments of accumulated motion compresses still frames hard and keeps
    articulation.  This is NOT the L2 boundary smoothing that Robotka et al. (2026,
    arXiv:2606.11925) found harmful: no frame is averaged toward its neighbour to look
    smoother -- only the sampling grid moves, and the duration distribution is untouched.

    Measured offline over 600 citation clips: closes 39% of the pause-density gap and 31% of
    the velocity gap to real.  It is a partial fix; the remainder is intrinsic citation-form
    articulation rate.
    """
    T = k.shape[0]
    if T < 3 or n_out < 2:
        return resample_clip(k, b, n_out)
    d = np.linalg.norm(np.diff(k[..., :2], axis=0), axis=2).mean(1)
    c = np.concatenate([[0.0], np.cumsum(d)])
    if not np.isfinite(c[-1]) or c[-1] <= 0:
        return resample_clip(k, b, n_out)
    idx = np.interp(np.linspace(0, 1, n_out), c / c[-1], np.arange(T))
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
    ap.add_argument("--drop_function_signs", action="store_true",
                    help="do not RENDER a sign for a span whose surface words are all stop words, or whose gloss is a "
                         "single fingerspelled letter. Real Auslan carries 0.467 signs per English word; the stitched "
                         "corpus carries 0.78 (09-05), because it signs the function words Auslan drops. The English "
                         "target is untouched, and the drop happens AFTER coverage/min_signs filtering, so the sentence "
                         "list is identical to a default build. A floor keeps the latest-positioned dropped spans when "
                         "fewer than --min_signs content spans remain.")
    ap.add_argument("--realizations", type=int, default=1)
    ap.add_argument("--max_clips", type=int, default=0, help="cap output (0 = no cap)")
    ap.add_argument("--trim_frac", type=float, default=0.15,
                    help="fraction of frames trimmed from each end of an isolated clip (rest pose)")
    ap.add_argument("--max_transition", type=int, default=4,
                    help="max interpolated transition frames between consecutive signs (0 = hard cut)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps_scale", type=float, default=1.0,
                    help="temporal rescale applied to every citation clip. Auslan-Daily is 25 fps "
                         "and MM-WLAuslan is 30 fps, so 0.8333 corrects a 1.2x dilation that was "
                         "previously baked into every synthetic clip (bug fix; 1.0 = old behaviour)")
    ap.add_argument("--sign_dur", action="store_true",
                    help="resample each sign to a length drawn from the REAL continuous per-sign "
                         "distribution (mean 22.5, median 20, p10 10, p90 37 frames @25fps) "
                         "instead of keeping the ~3-4x longer citation duration")
    ap.add_argument("--scramble_glosses", action="store_true",
                    help="DECISIVE CONTROL (proposed 2026-09-05, never built until now). Render each "
                         "sentence from RANDOMLY CHOSEN glosses instead of the ones it actually "
                         "contains, leaving the English target, the sign COUNT, the sampled per-sign "
                         "durations and the transition budgets identical. The only thing that changes "
                         "is WHICH sign is shown. If a scrambled corpus reproduces the joint arm's "
                         "gain, that gain is sequence structure / regularisation, not visual-lexical "
                         "grounding, and stitcher-fidelity work has no lexical headroom. If it falls "
                         "back to the real-only control, grounding is real and fidelity is worth "
                         "pursuing. Uses its OWN rng stream so durations/transitions are bit-matched.")
    ap.add_argument("--shuffle_order", action="store_true",
                    help="WAVE-10 CONTROL. Lay each sentence's signs out in a random (non-identity) order. "
                         "The English target, the exemplar drawn for each word, its duration, the number of "
                         "signs and the transition budget at each position are identical to the unshuffled "
                         "build; only the ORDER of the signs changes (default output is in English word order). "
                         "Uses its OWN rng stream.")
    ap.add_argument("--motion_resample", action="store_true",
                    help="with --sign_dur, resample each sign on a CUMULATIVE-MOTION timeline "
                         "rather than uniformly, so still frames compress and articulation is "
                         "kept (E0 2026-09-09). DEFAULT OFF: jobs are queued against the current "
                         "corpus and a default change would silently alter a running experiment.")
    ap.add_argument("--no_paren_alias", action="store_true",
                    help="do NOT index a gloss's parenthetical as a standalone alias -- it is "
                         "usually a sense disambiguator, so aliasing stitches wrong-sense signs")
    ap.add_argument("--letter_fix", action="store_true",
                    help="drop MM-WLAuslan's 24 fingerspelled single-letter glosses from the phrase index and "
                         "alias \"i\" to ME. Without it the English pronoun I is rendered as the LETTER I and "
                         "the article \"a\" as the LETTER A (8.67%% of segments in the trained 1.1:1 corpus).")
    ap.add_argument("--morphology", action="store_true",
                    help="let inflected words match citation-form glosses (must match Stage A)")
    ap.add_argument("--audit_jsonl", default="",
                    help="optional per-clip segment provenance for a motion-fidelity audit; this does "
                         "not change generated poses or the training manifest")
    a = ap.parse_args()
    if len(a.labels) != len(a.npz_dirs):
        raise SystemExit("--labels and --npz_dirs must be parallel lists")
    random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    clips_out = os.path.join(a.out_dir, "clips")
    os.makedirs(clips_out, exist_ok=True)

    lex = build_lexicon(a.labels, a.npz_dirs, morphology=a.morphology,
                        paren_alias=not a.no_paren_alias, letter_fix=a.letter_fix)
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
            source_frames_original = int(k.shape[0])
            if a.trim_frac > 0 and k.shape[0] > 8:          # drop rest pose at both ends
                cut = int(k.shape[0] * a.trim_frac)
                k, b = k[cut:k.shape[0] - cut], b[cut:b.shape[0] - cut]
            if len(cache) < 20000:                           # bounded LRU-ish cache
                cache[entry] = (k, b, source_frames_original)
            return k, b, source_frames_original
        return cache[entry]

    # separate stream: the main `random` gets exactly one choice() per sign either way, and numpy
    # `rng` is untouched, so durations and transition budgets are identical to the unscrambled build.
    scram = random.Random(a.seed + 1)
    shuf = random.Random(a.seed + 2)     # --shuffle_order's own stream, same reasoning
    all_glosses = sorted(lex.gloss2clips)

    rows, n_written, skipped = [], 0, 0
    dropped_signs = total_signs_before = floored = 0   # --drop_function_signs bookkeeping
    audit_fh = None
    if a.audit_jsonl:
        audit_dir = os.path.dirname(os.path.abspath(a.audit_jsonl))
        os.makedirs(audit_dir, exist_ok=True)
        audit_fh = open(a.audit_jsonl, "w", encoding="utf-8")
    sentences = [ln.strip() for ln in open(a.sentences, encoding="utf-8") if ln.strip()]
    print(f"[stitch2] {len(sentences)} candidate sentences", flush=True)

    for text in sentences:
        cov, spans = coverage(text, lex)
        if len(spans) < a.min_signs or cov < a.min_cov or not content_words(text):
            skipped += 1
            continue
        if a.drop_function_signs:
            toks = words(text)
            is_fn = [all(w in STOP for w in toks[i:j]) or len(g.strip()) == 1 for i, j, g in spans]
            kept = [sp for sp, f in zip(spans, is_fn) if not f]
            if len(kept) < a.min_signs:              # floor: top up with the latest-positioned dropped spans
                extra = [sp for sp, f in zip(spans, is_fn) if f][-(a.min_signs - len(kept)):]
                kept = sorted(kept + extra, key=lambda sp: sp[0])
                floored += 1
            dropped_signs += len(spans) - len(kept)
            total_signs_before += len(spans)
            spans = kept
        for _ in range(a.realizations):
            # pass 1: every random draw, in sentence order (exemplar, duration, transition budget), so
            # --shuffle_order changes nothing but the order the drawn signs are laid out in.
            drawn, budgets = [], []
            for si, (_, _, gloss) in enumerate(spans):
                if a.scramble_glosses:
                    gloss = scram.choice(all_glosses)
                source_dir, source_id = random.choice(lex.gloss2clips[gloss])
                k, b, source_frames_original = load((source_dir, source_id))
                source_frames_after_trim = int(k.shape[0])
                if a.sign_dur:                       # match the real continuous duration
                    _n = sample_sign_duration(rng)
                    k, b = (motion_resample_clip(k, b, _n) if a.motion_resample
                            else resample_clip(k, b, _n))
                elif a.fps_scale != 1.0:             # frame-rate correction only
                    k, b = resample_clip(k, b, max(4, int(round(k.shape[0] * a.fps_scale))))
                # transition BUDGET (sampled), not a smoothness objective: linear blend of the
                # two boundary poses. Robotka et al. (2026, arXiv:2606.11925) found that
                # optimising boundaries for L2 smoothness hurt SLT, so this stays short and randomised.
                budgets.append(int(rng.integers(0, a.max_transition + 1))
                               if si > 0 and a.max_transition > 0 else 0)
                drawn.append((gloss, source_dir, source_id, source_frames_original,
                              source_frames_after_trim, k, b))
            order = list(range(len(drawn)))
            if a.shuffle_order and len(order) > 1:   # uniform over the non-identity permutations
                while order == sorted(order):
                    shuf.shuffle(order)
            # pass 2: lay the signs out; a transition keeps its POSITION's budget and blends the new neighbours
            kps_parts, bbox_parts, segments = [], [], []
            cursor = 0
            for pos, oi in enumerate(order):
                (gloss, source_dir, source_id, source_frames_original,
                 source_frames_after_trim, k, b) = drawn[oi]
                transition_before = 0
                n_t = budgets[pos]
                if n_t and kps_parts:
                    prev, nxt = kps_parts[-1][-1], k[0]
                    w = np.linspace(0, 1, n_t + 2)[1:-1][:, None, None]
                    kps_parts.append((1 - w) * prev[None] + w * nxt[None])
                    bbox_parts.append(np.repeat(bbox_parts[-1][-1][None], n_t, axis=0))
                    transition_before = n_t
                    cursor += n_t
                seg_start = cursor
                cursor += int(k.shape[0])
                seg = {
                    "gloss": gloss,
                    "source_dir": source_dir,
                    "source_id": source_id,
                    "source_frames_original": source_frames_original,
                    "source_frames_after_trim": source_frames_after_trim,
                    "output_frames": int(k.shape[0]),
                    "start_frame": seg_start,
                    "end_frame": cursor - 1,
                    "transition_frames_before": transition_before,
                }
                if a.shuffle_order:
                    seg["english_position"] = oi
                segments.append(seg)
                kps_parts.append(k)
                bbox_parts.append(b)
            KPS = np.concatenate(kps_parts, 0).astype(np.float32)
            BBOX = np.concatenate(bbox_parts, 0).astype(np.float32)
            name = f"stitch2_{n_written:06d}.npz"
            np.savez_compressed(os.path.join(clips_out, name), KPS=KPS, BBOX=BBOX)
            clip_path = os.path.join(clips_out, name)
            rows.append(f"{clip_path}\t{text}")
            if audit_fh is not None:
                audit_fh.write(json.dumps({
                    "clip_path": clip_path,
                    "text": text,
                    "seed": a.seed,
                    "sign_dur": a.sign_dur,
                    "scramble_glosses": a.scramble_glosses,
                    "motion_resample": a.motion_resample,
                    "fps_scale": a.fps_scale,
                    "trim_frac": a.trim_frac,
                    "max_transition": a.max_transition,
                    "segments": segments,
                }, sort_keys=True) + "\n")
            n_written += 1
            if a.max_clips and n_written >= a.max_clips:
                break
        if a.max_clips and n_written >= a.max_clips:
            break

    man = os.path.join(a.out_dir, "posestitch_v2_train.tsv")
    open(man, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    if audit_fh is not None:
        audit_fh.close()
    if a.drop_function_signs:
        print(f"[stitch2] drop_function_signs: removed {dropped_signs} of {total_signs_before} spans "
              f"({100 * dropped_signs / max(total_signs_before, 1):.1f}%); {floored} sentences hit the min_signs floor",
              flush=True)
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
