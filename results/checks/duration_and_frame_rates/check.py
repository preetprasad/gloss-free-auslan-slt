"""Real-clip length and frame-rate facts behind the temporal-fidelity argument (appendix table of clip lengths).
Recomputed from per-clip frame counts of the shipped tensors, the alignment table and the sign-spotting table (see SOURCES.md)."""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]


def rows(name):
    return list(csv.DictReader(open(HERE / name)))


def col(name, key):
    return np.array([float(r[key]) for r in rows(name)])


def gate(ok, msg):
    print(f"GATE {msg}: {'ok' if ok else 'FAILED'}")
    if not ok:
        sys.exit("GATE FAILED")


# ---- gates: each input file is tied to an independent recorded value before anything is printed ----
bf, ef, bs, es = (col("alignment_frames.csv", k) for k in ("Begin_Frame", "End_Frame", "bs", "es"))
gate(round((ef - bf).mean(), 1) == 72.7, f"alignment-table mean End-Begin {(ef - bf).mean():.1f} frames, recorded 72.7")

cl = rows("clip_lengths.csv")
fr = np.array([float(r["frames"]) for r in cl])
pair = [(float(r["frames"]), float(r["align_len"])) for r in cl if r["align_len"]]
x, y = np.array(pair).T
within = (np.abs(x - y) <= 2).mean()
gate(len(fr) == 14041 and len(pair) > 12000 and within >= 0.99 and np.corrcoef(x, y)[0, 1] >= 0.999,
     f"per-clip tensor length vs its alignment span over {len(pair)} unique (video, subtitle) pairs: "
     f"{100 * within:.1f}% within 2 frames, r {np.corrcoef(x, y)[0, 1]:.4f}; recorded: all within 2 frames, r 1.0000")

spf = col("spotting_frames.csv", "End_Frame") - col("spotting_frames.csv", "Begin_Frame")
gate(len(spf) == 1069 and np.median(spf) == 20 and np.percentile(spf, 10) == 10 and np.percentile(spf, 90) == 37,
     f"sign-spotting table {len(spf)} instances, median {np.median(spf):.0f}, p10 {np.percentile(spf, 10):.0f}, "
     f"p90 {np.percentile(spf, 90):.0f}; recorded 1069, 20, 10, 37 (comment of the stitching builder)")

sup = json.load(open(ROOT / "results/stitched_corpus_audit/stitch_corpus_audit.json"))["visual_supply_adopted"]
mm = col("mmwl_lengths.csv", "frames")
ids = {r["id"]: float(r["frames"]) for r in rows("mmwl_lengths.csv")}
vf = {r["id"]: float(r["video_frames"]) for r in rows("mmwl_video_frames.csv")}
d = np.array([(vf[k], ids[k]) for k in vf])
gate(len(mm) == sup["mmwl_clips"] and len(vf) == sup["distinct_recordings_used"] and (np.abs(d[:, 0] - d[:, 1]) <= 1).mean() >= 0.99,
     f"dictionary pose file {len(mm)} clips; its frame counts agree within 1 frame with the counts extracted from the videos "
     f"for {len(vf)} recordings ({100 * (np.abs(d[:, 0] - d[:, 1]) <= 1).mean():.1f}%); recorded {sup['mmwl_clips']} clips, {sup['distinct_recordings_used']} recordings")

# ---- recomputed numbers ----
fps = (ef - bf).sum() / (es - bs).sum()
print(f"Auslan-Daily Communication frame rate {fps:.0f} fps (frames / seconds over {len(bf)} alignment rows)  [c15-017 c15-018]")
print(f"Real Auslan-Daily clip (a whole sentence, {fps:.0f} fps): mean {fr.mean():.1f} frames over {len(fr)} shipped tensors  [c15-017]")
print(f"Real continuous sign, per sign ({fps:.0f} fps): mean {spf.mean():.1f} frames over {len(spf)} sign-spotting instances (median {np.median(spf):.0f})  [c15-018]")

# ---- findings and half-reproduced claims, not tagged ----
print(f"Not recomputed: the 30 fps of MM-WLAuslan and the 1.2x it implies need the frame rate of the dictionary videos, which are not in this repository")
print(f"DISAGREES: MM-WLAuslan isolated-sign clip as shipped pose: mean {mm.mean():.1f} frames over {len(mm)} clips (dissertation: 79.4); "
      f"the video-extracted counts of {len(vf)} recordings give mean {d[:, 0].mean():.2f} against {d[:, 1].mean():.2f} for the same recordings in the pose file")
