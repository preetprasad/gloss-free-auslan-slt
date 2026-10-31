"""Published systems' own code (200 epochs) and this project's encoder with the published recipe.

Recomputes, from saved test-set hypotheses, the BLEU-4 scores, differences and 95% intervals that the
dissertation reports for (a) the authors' SL-Transformer / GASLT code on 53 DWPose points against the
project's translator trained in the same jobs (five seeds), (b) the project's translator trained with the
authors' recipe (three seeds), (c) the earlier runs on the first-listed AlphaPose person, and
(d) a few older single-run numbers that exist only as saved job summaries.

Run:  cd src && python3 ../results/checks/published_systems_200_epochs/check.py
"""
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import corpus_bleu4, detokenize  # noqa: E402

warnings.filterwarnings("ignore")
D = HERE / "data"
REC = json.loads((D / "recorded.json").read_text())


def rows(path):
    r = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(r) == 800, path
    if str(r[0]["ref"]).startswith("sha1:"):
        sys.exit("references are still digests: run `make data` first")
    return r


def bleu(path):
    r = rows(path)
    return corpus_bleu4([detokenize(x["hyp"]) for x in r], [detokenize(x["ref"]) for x in r])


# ---------------------------------------------------------------- scores
AUTH = {k: bleu(D / "runs" / (k.split("/")[0]) / (k.split("/")[1] + ".jsonl")) for k in REC["authors"]}
OWN = {}
for k in REC["own"]:
    if k.endswith(".b5") or k.endswith(".prot"):
        job, rest = k.split("/")
        OWN[k] = bleu(D / "own" / job / (rest + ".jsonl"))

# ---------------------------------------------------------------- gate
bad = [(k, v, REC["authors"][k]) for k, v in AUTH.items() if abs(v - REC["authors"][k]) > 1e-6]
bad += [(k, v, REC["own"][k]) for k, v in OWN.items() if abs(v - REC["own"][k]) > 1e-6]
print(f"GATE 1: {len(AUTH) + len(OWN)} saved test decodes re-scored through slt.metrics; "
      f"{len(AUTH) + len(OWN) - len(bad)} equal the score recorded by the job that wrote them (1e-6)")
if bad:
    print("GATE FAILED", bad[:5])
    sys.exit(1)


def ci(a, b):
    """Difference of means, pooled-variance two-sample t interval (95%)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    se = sp * math.sqrt(1 / n1 + 1 / n2)
    t = stats.t.ppf(0.975, n1 + n2 - 2)
    d = a.mean() - b.mean()
    return d, d - t * se, d + t * se


def f(x):
    return f"{x:+.2f}"


def show(label, a, b, per_seed=None, tag=""):
    d, lo, hi = ci(a, b)
    tail = ""
    if per_seed is not None:
        pos = sum(1 for x in per_seed if x > 0)
        tail = f"  seeds positive {pos} of {len(per_seed)}"
    print(f"  {label:<46} {f(d)} [{f(lo)}, {f(hi)}]{tail}  [{tag}]")
    return d, lo, hi


S3, S5 = (42, 101, 202), (42, 101, 202, 303, 404)
J = lambda s: "j167758" if s < 300 else "j179411"  # noqa: E731
EDW5 = [OWN[f"{J(s)}/Edw_s{s}.b5"] for s in S5]
EDW3 = EDW5[:3]

# gate 2: the three-seed Rule B differences printed by the 12l2 summary (job 167759)
sl3 = [AUTH[f"j167758/slt__dw53_frame_s{s}"] for s in S3]
ga3 = [AUTH[f"j167758/gaslt__dw53_frame_s{s}"] for s in S3]
g_slt, g_ga = ci(sl3, EDW3), ci(ga3, EDW3)
ok = all(abs(x - y) < 0.005 for x, y in zip(g_slt, (-1.91, -2.67, -1.15))) and \
    all(abs(x - y) < 0.006 for x, y in zip(g_ga, (-0.86, -1.77, 0.05)))
print("GATE 2: the three-seed Rule B differences of the 12l2 summary (SL-T -1.91, -2.67 to -1.15; "
      "GASLT -0.86, -1.77 to +0.05) are reproduced: " + ("yes" if ok else "NO"))
if not ok:
    sys.exit(1)

# ================================================================ 1. five seeds, authors' code on DWPose
print("\n== 1. The authors' code (200 epochs) on 53 DWPose points against this project's translator, five seeds "
      "(jobs 167758 + 179411), test BLEU-4, project scoring function")
sl5 = [AUTH[f"{J(s)}/slt__dw53_frame_s{s}"] for s in S5]
ga5 = [AUTH[f"{J(s)}/gaslt__dw53_frame_s{s}"] for s in S5]
print(f"  SL-Transformer, dw53 cells, {len(S5)} seeds  mean {np.mean(sl5):.2f}   per seed {[round(x, 2) for x in sl5]}  [c11-022]")
print(f"  GASLT, dw53 cells            mean {np.mean(ga5):.2f}   per seed {[round(x, 2) for x in ga5]}  [c11-022]")
print(f"  translator of this thesis (Edw, 40 epochs, beam 5) mean {np.mean(EDW5):.2f}   per seed {[round(x, 2) for x in EDW5]}  [c11-023]")
ep = [REC["edw_epochs"][f"{J(s)}/Edw_s{s}"] for s in S5]
print(f"  Edw epochs completed / planned: {ep}  [c11-023]")
print("  Rule B (authors' code minus translator, pooled t, 8 df):")
show("SL-Transformer - Edw", sl5, EDW5, tag="c11-024")
show("GASLT - Edw", ga5, EDW5, tag="c11-024")

# ================================================================ 2. 12l4
print("\n== 2. This project's translator with the published recipe (200 epochs, constant 5e-5), job 239443, three seeds")
T = "j239443"
arm = lambda a, kind: [OWN[f"{T}/{a}_s{s}.{kind}"] for s in S3]  # noqa: E731
O53b, O53p = arm("O53", "b5"), arm("O53", "prot")
Ocb, Edb = arm("Ocwb", "b5"), arm("Edw", "b5")
slt4 = [AUTH[f"{T}/slt__dw53_frame_s{s}"] for s in S3]
gas4 = [AUTH[f"{T}/gaslt__dw53_frame_s{s}"] for s in S3]
print(f"  O53  (our layers, 53 DWPose points)  authors' decoding protocol: mean {np.mean(O53p):.2f}   beam 5: mean {np.mean(O53b):.2f}  [c11-025 c19-073 c19-078]")
print(f"  Ocwb (our layers, usual input)       beam 5: mean {np.mean(Ocb):.2f}  [c11-027 c19-076]")
print(f"  Edw  (our recipe, 40 epochs)         beam 5: mean {np.mean(Edb):.2f}  [c11-027 c19-076]")
print(f"  authors' SL-Transformer, retrained in this job: mean {np.mean(slt4):.2f};  authors' GASLT: mean {np.mean(gas4):.2f}  [c11-025 c11-026 c19-073 c19-075]")
print("  O53 (authors' protocol) minus the authors' code retrained in the same job:")
show("O53 - SL-Transformer", O53p, slt4, [a - b for a, b in zip(O53p, slt4)], "c11-025 c19-073 c19-074")
show("O53 - GASLT", O53p, gas4, [a - b for a, b in zip(O53p, gas4)], "c11-026 c19-075")
print("  Recipe effect on this project's translator, beam 5:")
show("Ocwb - Edw", Ocb, Edb, [a - b for a, b in zip(Ocb, Edb)], "c11-027 c19-076 c19-077")
print("  Input format at the authors' recipe, beam 5:")
show("Ocwb - O53 (O53 beam-5 mean %.2f)" % np.mean(O53b), Ocb, O53b, [a - b for a, b in zip(Ocb, O53b)], "c19-078")
c = REC["cfg"][f"{T}/Ocwb_s42"]
print(f"  Ocwb settings: arrays of {c['num_keypoints']} keypoints, subset {c['keypoint_subset']}, "
      f"confidence {c['use_confidence']}, {c['coord_norm']} normalisation; {c['epochs']} epochs, "
      f"{c['lr_schedule']} learning rate {c['lr']:g}, batch {c['batch_size']}; decoded at beam width "
      f"{REC['own_beam'][T + '/Ocwb_s42']}  [c19-062]")

# ================================================================ 3. first-listed AlphaPose person runs
print("\n== 3. Earlier runs of the authors' code on the first-listed AlphaPose person (not the signer), test BLEU-4")
cells = {}
for k, v in AUTH.items():
    job, cell = k.split("/")
    name = cell.rsplit("_s", 1)[0]
    if "__ap" in name:
        cells.setdefault((job, name), []).append(v)
lab = {"j138567": "12l (stopped early)", "j167758": "12l2 (200 epochs)", "j179411": "w22 (200 epochs, 2 seeds)"}
allv = []
for (job, name), v in sorted(cells.items()):
    allv += v
    print(f"  {lab[job]:<26} {name:<22} mean {np.mean(v):.2f}  (n={len(v)})")
cm = [np.mean(v) for v in cells.values()]
print(f"  cell means range {min(cm):.2f} to {max(cm):.2f}; single runs range {min(allv):.2f} to {max(allv):.2f} "
      f"({len(allv)} runs, {len(cells)} cells)  [c19-056]")

# ================================================================ 4. older single runs: saved job outputs only
print("\n== 4. Older single runs. No per-item outputs (hypotheses) exist for them in this working tree: the folders "
      "of these runs hold a training log, a metrics.json or progress.json and a checkpoint only, so the scores are read, not recomputed")
O = D / "old"
J_ = lambda n: json.loads((O / n).read_text())  # noqa: E731


def last_epoch(log):
    ep = [int(l.split("]")[0].split()[1]) for l in open(O / log) if l.startswith("[ep ")]
    return ep[-1]


ex = J_("exact_slt_run_summary.json")
t = ex["selected_checkpoint_decode"]["test"]
print(f"  exact SL-Transformer re-implementation ({ex['model']['parameters']:,} parameters), distributed keypoints, official test: "
      f"beam 5 {t['beam5_bleu4']:.2f}, greedy {t['greedy_bleu4']:.2f}  "
      f"(read from the saved run summary exact_slt_run_summary.json; its training log ends at epoch {last_epoch('exact_slt_train.log')} "
      f"of {ex['training']['planned_epochs']} and exact_slt_progress.json says {J_('exact_slt_progress.json')['status']})  [c08-016]")
sj = J_("signjoey_run_summary.json")
print(f"  25.9 M SignJoey-compatible model ({sj['architecture']['parameters']:,} parameters), first-listed AlphaPose keypoints: "
      f"greedy test {sj['best_checkpoint_test']['greedy_bleu4_13a_detokenized']:.2f}  "
      f"(read from the saved run summary signjoey_run_summary.json; its training log ends at epoch {last_epoch('signjoey_train.log')} "
      f"of {sj['training']['planned_epochs']})  [c16-021]")
M = {k: J_(f"dwpose_{k}.metrics.json") for k in ("d512", "full")}
for k, nm in (("d512", "d=512, 6+6 layers on DWPose"), ("full", "default size on DWPose")):
    m = M[k]
    cf = m["config"]
    print(f"  {nm}: greedy test {m['test']['bleu4']:.2f}  (d_model {cf['d_model']}, {cf['num_encoder_layers']}+{cf['num_decoder_layers']} layers; "
          f"read from the metrics.json that the job wrote, best epoch {m['best']['epoch']})  [c16-021]")
print("  recorded training settings of the runs in this row of the appendix table (c16-022 compares them):")
for k in ("d512", "full"):
    cf = M[k]["config"]
    print(f"    {k} DWPose run, test {M[k]['test']['bleu4']:.2f} (metrics.json of the job): {cf['epochs']} epochs, {cf['lr_schedule']} "
          f"learning rate {cf['lr']:g}, batch {cf['batch_size']} x {cf['grad_accum_steps']} accumulation")
C = J_("checkpoint_cfgs.json")
sc = C["signjoey_reference"]
print(f"    25.9 M model scored {sj['best_checkpoint_test']['greedy_bleu4_13a_detokenized']:.2f} (checkpoint settings): {sc['epochs']} epochs planned (ran {last_epoch('signjoey_train.log')}), "
      f"{sc['lr_schedule']} learning rate {sc['lr']:g}, batch {sc['batch_size']} x {sc['grad_accum_steps']}")
a2 = J_("accum2_metrics.json")
c2 = a2["config"]
print(f"    another 25.9 M-class model (512-d, 3+3 layers, 1024-d adapter, first-listed AlphaPose; not the {sj['best_checkpoint_test']['greedy_bleu4_13a_detokenized']:.2f} run): greedy test "
      f"{a2['test']['bleu4']:.2f} (metrics.json of the job), {c2['epochs']} epochs, {c2['lr_schedule']} learning rate {c2['lr']:g}, "
      f"batch {c2['batch_size']} x {c2['grad_accum_steps']}")
d5 = M["d512"]["config"]
print("  c16-022 CHECK: the 7.11 run recorded " + f"{d5['epochs']} epochs at {d5['lr']:g} (same as the default-size run: "
      f"{M['full']['config']['epochs']} at {M['full']['config']['lr']:g}); '60 epochs at 3e-4' is recorded only for the 2.29 run above: "
      + ("DISAGREES with the dissertation sentence for the 7.11 run" if (d5["epochs"], d5["lr"]) != (60, 3e-4) else "agrees"))
