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
import argparse, csv, json, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.stitch_lexicon import build_lexicon, coverage, content_words, words, STOP, WRONG_SENSE_GLOSSES
CUT_THR = 0.8    # wave 16 default of --w16_cut_thr (equals tools/w16_posture.CUT_THR, asserted when the flag is used); tools.w16_posture is imported only with --w16_restcut


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


def scale_to_total(lengths, total: int):
    """Integer lengths proportional to `lengths` that sum EXACTLY to `total` (largest remainder; ties by position).
    Used by --match_frames. Every length must stay >= 1. Integer arithmetic: float remainders broke exact ties
    (e.g. 8/19 vs 8/19) arbitrarily on 44 of 13,760 clips in prep job 66604, which its gate caught."""
    s = sum(lengths)
    out = [n * total // s for n in lengths]
    rem = [n * total % s for n in lengths]
    for i in sorted(range(len(rem)), key=lambda i: (-rem[i], i))[:total - sum(out)]:
        out[i] += 1
    if sum(out) != total or min(out) < 1:
        raise SystemExit(f"--match_frames cannot allocate {total} frames over sign lengths {lengths}")
    return out


def _manifest_clips(path):
    """Clip names (file name without .npz) of a train/val/test manifest (`path<TAB>text` rows)."""
    return {os.path.basename(l.split("\t", 1)[0])[:-4] for l in open(path, encoding="utf-8") if "\t" in l}


def load_inreg(csv_path, npz_dir, train_manifest, exclude_manifests, min_frames=4):
    """WAVE-15 in-register exemplars: {gloss: [(npz_dir, str(Video_ID)), ...]} (sorted by Video_ID) and a stats dict.

    Source: results/wave12f_out/located.csv = Auslan-Daily's isolated-sign clips, each located by pixels inside ONE Communication
    signer clip (wave 12f). Kept: status == located (the 24 ambiguous and 2 empty ones go), comm_split == train, the source clip is
    in the TRAIN manifest and in none of the exclude manifests (validation, test), and the clip has >= min_frames frames (the builder
    floors a sign at 4). The pose files of the kept exemplars of glosses the lexicon also has must exist (checked in main): a missing file stops the build."""
    train, held = _manifest_clips(train_manifest), set()
    for m in exclude_manifests:
        held |= _manifest_clips(m)
    st = dict(rows=0, located=0, not_located=0, comm_split_not_train=0, in_held=0, not_in_train=0, short=0, kept=0)
    out = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        st["rows"] += 1
        if r["status"] != "located":
            st["not_located"] += 1; continue
        st["located"] += 1
        clip = r["Video_Name_Clip"]
        if clip in held:
            st["in_held"] += 1; continue
        if r["comm_split"] != "train":
            st["comm_split_not_train"] += 1; continue
        if clip not in train:
            st["not_in_train"] += 1; continue
        if int(r["frames"]) < min_frames:
            st["short"] += 1; continue
        out.setdefault(r["Query_Gloss"].strip().lower(), []).append((npz_dir, r["Video_ID"]))
        st["kept"] += 1
    return {g: sorted(v, key=lambda e: int(e[1])) for g, v in out.items()}, st


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
    ap.add_argument("--match_frames", default="",
                    help="WAVE-7 STAGE-2 DL ARM. A reference manifest (posestitch_v2_train.tsv of the arm-A corpus). Each "
                         "output clip's per-sign lengths are scaled, in proportion to the lengths --sign_dur sampled, so the "
                         "clip's TOTAL frame count (signs + transitions) equals the clip in the SAME row of the reference. "
                         "The English text must match row for row. No extra randomness is drawn, so the exemplars, transition "
                         "budgets and sign order are those of the same build without this flag; only per-sign lengths change. "
                         "Needs --sign_dur and --realizations 1.")
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
    ap.add_argument("--drop_wrong_sense", action="store_true",
                    help="WAVE-12b W ARM. Do not RENDER a sign whose gloss is one of the 14 in "
                         "stitch_lexicon.WRONG_SENSE_GLOSSES (\"be\" -> BE (NOT BE), \"can\" -> CAN (TIN), ...: 3,257 of "
                         "94,428 signs of the adopted corpus). The sentence list, the English text and EVERY random draw "
                         "are those of the same build without the flag, so each sign that stays is the same exemplar at the "
                         "same length; a transition keeps the budget drawn for the sign it precedes. A clip whose signs are "
                         "all wrong-sense is left as built. Needs --no_paren_alias (with aliases on, \"not be\" is a "
                         "correct match for the same gloss).")
    ap.add_argument("--inreg_csv", default="",
                    help="WAVE-15 IN-REGISTER ARM. results/wave12f_out/located.csv. For every sign whose gloss label equals a gloss that has "
                         "in-register exemplars (Auslan-Daily isolated clips located inside a TRAIN clip, wave 12f), the exemplar is drawn "
                         "from those instead of MM-WLAuslan, with a SEPARATE random.Random(seed + 3). The main exemplar draw (one "
                         "random.choice per sign), the sampled durations and the transition budgets are exactly those of the build without "
                         "the flag; the in-register source is not end-trimmed (it already is the sign span) but IS resampled to the same "
                         "sampled length, so every clip and segment has the length it has without the flag. Needs --sign_dur, "
                         "--inreg_npz_dir and --inreg_train_manifest.")
    ap.add_argument("--inreg_npz_dir", default="", help="dwpose_isolated/ (<Video_ID>.npz, identical to the production DWPose frames)")
    ap.add_argument("--inreg_train_manifest", default="", help="the real Communication TRAIN manifest")
    ap.add_argument("--inreg_exclude_manifests", nargs="*", default=[],
                    help="validation and test manifests: an exemplar whose source clip is in one of them is dropped")
    ap.add_argument("--w16_restcut", action="store_true",
                    help="WAVE-16 REST-RETURN ARM. Before an exemplar (after the --trim_frac trim, exactly as in the build without the flag) is "
                         "resampled to its sampled length, cut its leading and trailing frames in which BOTH hands are down (tools/w16_posture.py: wrist "
                         "more than --w16_cut_thr shoulder widths below the shoulder midpoint, or wrist confidence under 0.3), unless fewer than 4 frames would "
                         "remain or no frame has a hand up (then the sign is left as it is). The remaining frames are resampled to the SAME sampled "
                         "length, so every segment and clip has the length it has without the flag. No random number is drawn: exemplars, durations and "
                         "transition budgets are those of the build without the flag. Needs --sign_dur. The audit sidecar gains restcut_lead, "
                         "restcut_trail and source_frames_after_restcut per segment (only with the flag).")
    ap.add_argument("--w16_cut_thr", type=float, default=CUT_THR,
                    help="with --w16_restcut: a frame is cut at a sign edge if both wrists are more than this many shoulder widths below the shoulder "
                         "midpoint (or low-confidence). Default 0.8 (a margin of 0.2 under the 1.0 of the measurement, see tools/w16_posture.edge_cut); 1.0 is the literal definition.")
    ap.add_argument("--audit_jsonl", default="",
                    help="optional per-clip segment provenance for a motion-fidelity audit; this does "
                         "not change generated poses or the training manifest")
    a = ap.parse_args()
    if len(a.labels) != len(a.npz_dirs):
        raise SystemExit("--labels and --npz_dirs must be parallel lists")
    if a.drop_wrong_sense and (not a.no_paren_alias or a.scramble_glosses or a.shuffle_order or a.match_frames
                               or a.drop_function_signs):
        raise SystemExit("--drop_wrong_sense needs --no_paren_alias and is not built to combine with "
                         "--scramble_glosses/--shuffle_order/--match_frames/--drop_function_signs")
    if a.inreg_csv:
        if not (a.inreg_npz_dir and a.inreg_train_manifest and a.sign_dur):
            raise SystemExit("--inreg_csv needs --inreg_npz_dir, --inreg_train_manifest and --sign_dur (lengths must equal the build without it)")
        if a.scramble_glosses or a.shuffle_order or a.match_frames or a.drop_function_signs or a.drop_wrong_sense or a.motion_resample:
            raise SystemExit("--inreg_csv is not built to combine with --scramble_glosses/--shuffle_order/--match_frames/"
                             "--drop_function_signs/--drop_wrong_sense/--motion_resample")
    if a.w16_restcut:
        if not a.sign_dur:
            raise SystemExit("--w16_restcut needs --sign_dur (lengths must equal the build without it)")
        if a.inreg_csv or a.scramble_glosses or a.shuffle_order or a.match_frames or a.drop_function_signs or a.drop_wrong_sense or a.motion_resample:
            raise SystemExit("--w16_restcut is not built to combine with --inreg_csv/--scramble_glosses/--shuffle_order/--match_frames/"
                             "--drop_function_signs/--drop_wrong_sense/--motion_resample")
        from tools.w16_posture import edge_cut, CUT_THR as _CT      # imported only with the flag: a snapshot without w16_posture.py still runs every other build
        assert _CT == CUT_THR, "builder CUT_THR default differs from tools/w16_posture.CUT_THR"
    ref_rows = None
    if a.match_frames:
        if not a.sign_dur or a.realizations != 1 or a.shuffle_order or a.scramble_glosses:
            raise SystemExit("--match_frames needs --sign_dur and --realizations 1, without --shuffle_order/--scramble_glosses")
        ref_rows = [ln.rstrip("\n").split("\t", 1) for ln in open(a.match_frames, encoding="utf-8") if "\t" in ln]
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

    inreg, n_inreg_signs = {}, 0
    if a.inreg_csv:
        inreg, ist = load_inreg(a.inreg_csv, a.inreg_npz_dir, a.inreg_train_manifest, a.inreg_exclude_manifests)
        inreg = {g: v for g, v in inreg.items() if g in lex.gloss2clips}       # only glosses the MM lexicon also has
        miss = [e[1] for v in inreg.values() for e in v if not os.path.isfile(os.path.join(e[0], e[1] + ".npz"))]
        if miss:                                                               # a missing pose file stops the build instead of shrinking it
            raise SystemExit(f"--inreg_npz_dir: {len(miss)} pose files are missing, e.g. {miss[:3]}")
        print(f"[stitch2] in-register table: {ist} -> {len(inreg)} glosses shared with the lexicon, "
              f"{sum(len(v) for v in inreg.values())} exemplars", flush=True)

    cache = {}

    def load(entry):
        if entry not in cache:
            d, cid = entry
            z = np.load(os.path.join(d, cid + ".npz"))
            k, b = z["KPS"].astype(np.float32), z["BBOX"].astype(np.float32)
            source_frames_original = int(k.shape[0])
            if a.trim_frac > 0 and k.shape[0] > 8 and not (a.inreg_csv and d == a.inreg_npz_dir):   # drop rest pose at both ends (not for an in-register span)
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
    inreg_rng = random.Random(a.seed + 3)   # --inreg_csv's own stream: the main `random` is untouched by the swap
    all_glosses = sorted(lex.gloss2clips)

    rows, n_written, skipped = [], 0, 0
    dropped_signs = total_signs_before = floored = 0   # --drop_function_signs bookkeeping
    ws_removed = ws_total = ws_whole = 0               # --drop_wrong_sense bookkeeping
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
            drawn, budgets, sampled, cuts = [], [], [], []
            for si, (_, _, gloss) in enumerate(spans):
                if a.scramble_glosses:
                    gloss = scram.choice(all_glosses)
                source_dir, source_id = random.choice(lex.gloss2clips[gloss])
                if gloss in inreg:                   # --inreg_csv: same draw as above, then the source is replaced
                    source_dir, source_id = inreg_rng.choice(inreg[gloss])
                    n_inreg_signs += 1
                k, b, source_frames_original = load((source_dir, source_id))
                source_frames_after_trim = int(k.shape[0])
                if a.w16_restcut:                    # posture cut on the trimmed exemplar; deterministic, no random draw
                    lead, trail = edge_cut(k, thr=a.w16_cut_thr)
                    if lead or trail:
                        k, b = k[lead:k.shape[0] - trail], b[lead:b.shape[0] - trail]
                    cuts.append((lead, trail, int(k.shape[0])))
                if a.sign_dur:                       # match the real continuous duration
                    _n = sample_sign_duration(rng)
                    sampled.append(_n)
                    if not a.match_frames:           # with --match_frames: resampled after pass 1, once rescaled
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
            ref_total = None
            if a.match_frames:                       # DL arm: same draws, sign lengths rescaled to the reference row
                ref_path, ref_text = ref_rows[n_written]
                if ref_text != text:
                    raise SystemExit(f"--match_frames row {n_written}: reference text {ref_text!r} != {text!r}")
                ref_total = int(np.load(ref_path)["KPS"].shape[0])
                lens = scale_to_total(sampled, ref_total - sum(budgets))
                rs = motion_resample_clip if a.motion_resample else resample_clip
                drawn = [d[:5] + rs(d[5], d[6], n) for d, n in zip(drawn, lens)]
            if a.drop_wrong_sense:                   # after pass 1, so no draw above depends on the flag
                keep = [i for i, d in enumerate(drawn) if d[0] not in WRONG_SENSE_GLOSSES]
                ws_total += len(drawn)
                if keep:
                    ws_removed += len(drawn) - len(keep)
                    drawn, budgets = [drawn[i] for i in keep], [budgets[i] for i in keep]
                else:                                # every sign is wrong-sense: an empty clip cannot be written
                    ws_whole += 1
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
                if a.inreg_csv:
                    seg["inreg"] = source_dir == a.inreg_npz_dir
                if a.w16_restcut:                    # order is the identity here (not combined with --shuffle_order / --drop_wrong_sense)
                    seg["restcut_lead"], seg["restcut_trail"], seg["source_frames_after_restcut"] = cuts[oi]
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
                rec = {
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
                }
                if a.match_frames:
                    rec["match_frames_total"] = ref_total
                    rec["sampled_frames"] = sampled
                audit_fh.write(json.dumps(rec, sort_keys=True) + "\n")
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
    if a.drop_wrong_sense:
        print(f"[stitch2] drop_wrong_sense: removed {ws_removed} of {ws_total} signs "
              f"({100 * ws_removed / max(ws_total, 1):.2f}%); {ws_whole} clips left as built (all signs wrong-sense)",
              flush=True)
    if a.inreg_csv:
        print(f"[stitch2] inreg: {n_inreg_signs} signs drew an in-register exemplar", flush=True)
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
