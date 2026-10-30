#!/usr/bin/env python3
"""G1 readout: cross-modal ensembling on the video-disjoint (VD) split (results/PREREG_VD_XMODAL_2026-09-25.md).

Computes, from saved decode dumps only (no model is run):
  gates   - vd_val 1,444 / vd_test 1,254 rows with 0 clip overlap; every dump's references equal its manifest text in
            order; every dump re-scores through slt/metrics.py to the BLEU-4 and WER printed in its log (exact at the
            log's printed precision: 3 decimals BLEU, 4 decimals WER); the t2-vdens dumps re-score to 7.123 (V6) /
            7.807 (V3B) / 7.287 (vdB_s101, the t2-vdens best single); the RGB guard printed its feature-file count in
            every fusion-containing decode; each new fusion member's warm start transferred 129/132 base tensors with
            only the three vocabulary-sized tensors differing.
  selection - each swept system's LP = its vd_val argmax at full precision, first maximum in grid order (the rule
            seedens_analysis.py used); its vd_test row is read at that LP. V6/V3A/V3B were decoded on vd_test at one
            LP only (t2-vdens chose it from its vd_val log); that LP is re-derived here from the vd_val log.
  baseline - best-on-vd_val single among VD5's five members (vdB s42/101/202 from seed-ensemble job 21736, fus_s303 /
            fus_s404 from job 31205).
  C1      - tools/holm_decision.py --maxt (unmodified CLI): max-T simultaneous 95% intervals for V6, V3A, V3B, vdB ens5
            and VD5, each minus the baseline, 2,000 paired sentence resamples, seed 0.
  C2      - tools/paired_bootstrap.py (unmodified CLI): VD5 - V3B, 2,000 resamples, seed 0.
  claim   - mapped to the pre-registered wording.
  secondary (descriptive) - unseen-target BLEU (reference never in vd_train text, lowercased, whitespace-collapsed),
            content precision/recall and length ratio via tools/arm_readouts.readouts, VD5 vs V3B and vs baseline;
            plus tools/paired_bootstrap.py --subset unseen intervals.

Run:  ./.venv/bin/python results/video_disjoint_cross_modal_ensemble/g1_readout.py | tee results/video_disjoint_cross_modal_ensemble/g1_readout.txt
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PB = os.path.join(ROOT, "src")
TOOLS = os.path.join(PB, "tools")
sys.path[:0] = [PB, TOOLS]
from slt.metrics import corpus_bleu4, corpus_wer, detokenize  # noqa: E402
from arm_readouts import readouts                              # noqa: E402
from stitch_lexicon import STOP                                # noqa: E402

LPS = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
DEC, SE = os.path.join(HERE, "decode"), os.path.join(HERE, "prior", "seedens_vdB")
T2 = os.path.join(ROOT, "results", "t2_vdens_10674")          # md5-identical to Kaya /group/.../t2_vdens (checked)
T2VAL = os.path.join(HERE, "prior", "t2_vdens_val")
MAN = os.path.join(HERE, "manifests")
norm = lambda t: " ".join(t.lower().split())


def manifest(split):
    rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(MAN, f"vd_{split}.tsv"), encoding="utf-8") if "\t" in l]
    return [c for c, _ in rows], [t for _, t in rows]


def load(path):
    rows = sorted((json.loads(l) for l in open(path, encoding="utf-8")), key=lambda r: r["idx"])
    assert [r["idx"] for r in rows] == list(range(len(rows))), path
    return [r["hyp_beam"] for r in rows], [r["ref"] for r in rows]


def logged(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        m = re.match(r"\s+(\S+)\s+BLEU-4=\s*([\d.]+)\s+WER=([\d.]+)", l)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3))
    assert out, f"no score lines in {path}"
    return out


def gate_dir(d, log, texts):
    """Re-score every hyps_*.jsonl in d against its log line; return {tag: (bleu, raw_hyps)}."""
    lg = logged(log)
    files = {re.match(r"hyps_(.+)\.jsonl$", os.path.basename(f)).group(1): f for f in glob.glob(os.path.join(d, "hyps_*.jsonl"))}
    assert set(files) == set(lg), f"{d}: dumps {sorted(files)} vs log {sorted(lg)}"
    out = {}
    for tag, f in files.items():
        h, r = load(f)
        assert [x.strip() for x in r] == [t.strip() for t in texts], f"{f}: references differ from the manifest"
        H, R = [detokenize(x) for x in h], [detokenize(x) for x in r]
        b, w = corpus_bleu4(H, R, "13a"), corpus_wer(H, R)
        assert (f"{b:.3f}", f"{w:.4f}") == lg[tag], f"{f}: re-scores {b:.3f}/{w:.4f}, log {lg[tag]}"
        out[tag] = (b, h)
    return out


def argmax_lp(scores):
    return max(LPS, key=lambda x: (scores[x], -LPS.index(x)))      # first maximum in grid order


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    py = sys.executable
    # ---------------------------------------------------------------- gate: manifests
    vc, vt = manifest("val"); tc, tt = manifest("test"); trc, trt = manifest("train")
    assert (len(vc), len(tc), len(trc)) == (1444, 1254, 11341), (len(vc), len(tc), len(trc))
    assert not set(vc) & set(tc) and not set(trc) & (set(vc) | set(tc)), "clip overlap between VD splits"
    print(f"[gate] vd_train {len(trc)} / vd_val {len(vc)} / vd_test {len(tc)} rows; clip overlap 0 (val-test, train-val/test)")
    for s in ("train", "val", "test"):
        print(f"       vd_{s}.tsv md5 {md5(os.path.join(MAN, f'vd_{s}.tsv'))}")
    texts = {"val": vt, "test": tt}

    # ---------------------------------------------------------------- gate: warm start (job 31204)
    L = os.path.join(HERE, "logs")
    for s in (303, 404):
        tl = open(os.path.join(L, f"fus_s{s}.train.log")).read()
        jo = open(os.path.join(L, f"vd-fusion-31204_{0 if s == 303 else 1}.out")).read()
        assert f"[init] warm-started base with 129/132 tensors from /group/pmc099/pprasad/wave3_vd/runs/vdB_s{s}/best.pt" in tl
        assert ("[gate] warm-started 129/132; shape-differing base tensors 3, of which vocab-sized 3: "
                "['tok_emb.weight', 'generator.weight', 'generator.bias']") in jo and "[gate] OK" in jo
        assert f"VD_FUSION_DONE s{s}" in jo and "RGB features missing for 0" in jo
        print(f"[gate] fus_s{s}: warm-started 129/132 from vdB_s{s}; only tok_emb.weight, generator.weight, "
              f"generator.bias (2,867 -> 2,805 rows) re-initialised")

    # ---------------------------------------------------------------- gate: RGB guard (job 31205)
    jo = open(os.path.join(L, "vd-xmodal-31205.out")).read()
    assert "vd_val 1444 / vd_test 1254 / overlap 0" in jo and "VD_XMODAL_DECODE_DONE" in jo
    for n in ("fus_s303", "fus_s404", "VD5"):
        for sp in ("val", "test"):
            lg = open(os.path.join(DEC, f"{n}_{sp}.log")).read()
            assert "rgb_feat_dir=/group/pmc099/pprasad/scratch_backup/dwpose/rgb_feats (14041 feature files)" in lg, (n, sp)
            assert "[preproc] OK" in lg, (n, sp)
    print("[gate] RGB guard printed '(14041 feature files)' and the preprocessing guard passed in all 6 decodes of 31205")

    # ---------------------------------------------------------------- gate: every swept dump re-scores to its log
    swept = {"vdB_s42": (SE, "s42"), "vdB_s101": (SE, "s101"), "vdB_s202": (SE, "s202"),
             "fus_s303": (DEC, "fus_s303"), "fus_s404": (DEC, "fus_s404"), "vdB_ens5": (SE, "ens5"), "VD5": (DEC, "VD5")}
    S, nfiles = {}, 0
    for name, (base, stem) in swept.items():
        S[name] = {sp: gate_dir(os.path.join(base, f"{stem}_{sp}"), os.path.join(base, f"{stem}_{sp}.log"), texts[sp])
                   for sp in ("val", "test")}
        nfiles += sum(len(v) for v in S[name].values())
    print(f"[gate] {nfiles} swept dumps (LP 0.2..1.0 + MBR-WER + MBR-chrF, val and test, 7 systems) re-score to their "
          f"logs exactly at printed precision; references equal the manifests in order")

    # ---------------------------------------------------------------- gate: t2-vdens dumps
    T = {}
    for name, stem in (("V6", "V6"), ("V3A", "V3A"), ("V3B", "V3B"), ("t2_vdB_s101", "vdB_s101")):
        val = {k: float(v[0]) for k, v in logged(os.path.join(T2VAL, f"{stem}_val.log")).items() if k.startswith("LP=")}
        lp = max(LPS, key=lambda x: (val[f"LP={x}"], -LPS.index(x)))   # t2-vdens rule: first max of the 3-dp log values
        g = gate_dir(os.path.join(T2, f"dumps_{stem}_test"), os.path.join(T2, f"{stem}_test.log"), tt)
        assert f"LP={lp}" in g, f"{name}: test dump LP {sorted(g)} is not its vd_val-log argmax {lp}"
        T[name] = dict(lp=lp, val=val[f"LP={lp}"], test=g[f"LP={lp}"][0], hyps=g[f"LP={lp}"][1],
                       path=os.path.join(T2, f"dumps_{stem}_test", f"hyps_LP={lp}.jsonl"))
    for name, rec in (("V6", "7.123"), ("V3B", "7.807"), ("t2_vdB_s101", "7.287"), ("V3A", "6.531")):
        assert f"{T[name]['test']:.3f}" == rec, (name, T[name]["test"], rec)
    print(f"[gate] t2-vdens vd_test dumps re-score: V6 {T['V6']['test']:.3f} @LP{T['V6']['lp']} (rec 7.123), "
          f"V3B {T['V3B']['test']:.3f} @LP{T['V3B']['lp']} (rec 7.807), best single vdB_s101 {T['t2_vdB_s101']['test']:.3f} "
          f"@LP{T['t2_vdB_s101']['lp']} (rec 7.287); V3A {T['V3A']['test']:.3f} @LP{T['V3A']['lp']} (rec 6.531). "
          f"Each test LP = the vd_val-log argmax.")

    # ---------------------------------------------------------------- selection
    sel = {}
    for name in swept:
        v = {x: S[name]["val"][f"LP={x}"][0] for x in LPS}
        lp = argmax_lp(v)
        sel[name] = dict(lp=lp, val=v[lp], test=S[name]["test"][f"LP={lp}"][0], hyps=S[name]["test"][f"LP={lp}"][1],
                         path=os.path.join(swept[name][0], f"{swept[name][1]}_test", f"hyps_LP={lp}.jsonl"))
    # record check: job 21736's analysis (results/seedens_21736/analysis.txt, "[VD descriptive]") printed these
    for n, rec in (("vdB_s42", ("6.600", "7.407")), ("vdB_s101", ("6.718", "7.287")), ("vdB_s202", ("6.373", "7.149")),
                   ("vdB_ens5", ("6.552", "7.844"))):
        assert (f"{sel[n]['val']:.3f}", f"{sel[n]['test']:.3f}") == rec, (n, sel[n]["val"], sel[n]["test"], rec)
    print("[gate] vdB s42/101/202 and ens5 selections reproduce job 21736's recorded val/test values")
    print("\nSelection: LP = vd_val argmax (full precision, first in grid); vd_test read at that LP. Beam 10.")
    print(f"  {'system':<10}{'LP':>5}{'vd_val':>9}{'vd_test':>9}   members")
    mem = {"VD5": "vdB s42/101/202 + fus_s303 + fus_s404", "vdB_ens5": "vdB s42/101/202/303/404",
           "V6": "vdA s42/101/202 + vdB s42/101/202", "V3A": "vdA s42/101/202", "V3B": "vdB s42/101/202"}
    for name in swept:
        print(f"  {name:<10}{sel[name]['lp']:>5}{sel[name]['val']:>9.3f}{sel[name]['test']:>9.3f}   {mem.get(name, 'single')}")
    for name in ("V6", "V3A", "V3B"):
        print(f"  {name:<10}{T[name]['lp']:>5}{T[name]['val']:>9.3f}{T[name]['test']:>9.3f}   {mem[name]}  "
              f"(t2-vdens 10674; vd_val from its log, 3 dp)")
    singles = ["vdB_s42", "vdB_s101", "vdB_s202", "fus_s303", "fus_s404"]
    base = max(singles, key=lambda n: sel[n]["val"])
    B = sel[base]
    print(f"\nBaseline (best-on-vd_val single among VD5's members): {base} @LP{B['lp']}  vd_val {B['val']:.3f}  "
          f"vd_test {B['test']:.3f}")
    same = sum(a == b for a, b in zip(B["hyps"], T["t2_vdB_s101"]["hyps"]))
    print(f"  (descriptive: its CPU dump (job 21736) and the GPU dump of the same checkpoint in t2-vdens share "
          f"{same}/{len(tt)} identical hypotheses at the same LP; both score {T['t2_vdB_s101']['test']:.3f} / {B['test']:.3f})")

    print("\nvd_val deltas (reported, NOT a gate):")
    for name in ("V6", "V3A", "V3B"):
        print(f"  {name:<9} - {base}: {T[name]['val'] - B['val']:+.3f}")
    print(f"  vdB_ens5  - {base}: {sel['vdB_ens5']['val'] - B['val']:+.3f}")
    print(f"  VD5       - {base}: {sel['VD5']['val'] - B['val']:+.3f}")
    print(f"  VD5       - V3B     : {sel['VD5']['val'] - T['V3B']['val']:+.3f}")
    print(f"  VD5       - vdB_ens5: {sel['VD5']['val'] - sel['vdB_ens5']['val']:+.3f}")
    print("vd_test point deltas:")
    for name in ("V6", "V3A", "V3B"):
        print(f"  {name:<9} - {base}: {T[name]['test'] - B['test']:+.3f}")
    print(f"  vdB_ens5  - {base}: {sel['vdB_ens5']['test'] - B['test']:+.3f}")
    print(f"  VD5       - {base}: {sel['VD5']['test'] - B['test']:+.3f}")
    print(f"  VD5       - V3B     : {sel['VD5']['test'] - T['V3B']['test']:+.3f}")
    print(f"  VD5       - vdB_ens5: {sel['VD5']['test'] - sel['vdB_ens5']['test']:+.3f}")
    sys.stdout.flush()

    # ---------------------------------------------------------------- C1: max-T (holm_decision.py CLI, unmodified)
    looks = [(base, B), ("V6", T["V6"]), ("V3A", T["V3A"]), ("V3B", T["V3B"]),
             ("vdB_ens5", sel["vdB_ens5"]), ("VD5", sel["VD5"])]
    cmd = [py, os.path.join(TOOLS, "holm_decision.py"), "--maxt", "--baseline", base, "--n", "2000", "--seed", "0",
           "--alpha", "0.05"]
    for n, d in looks:
        cmd += ["--system", f"{n}={d['path']}", "--expect", f"{n}={d['test']:.3f}"]
    print("\n=== C1: " + " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd))
    out1 = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    print(out1, end="")
    m = re.search(r"^\s+VD5\s+([+-][\d.]+)\s+([\d.]+)\s+\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]\s*\*?\s+"
                  r"\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]", out1, re.M)
    assert m, "could not parse the VD5 row of the max-T output"
    c1_obs, c1_lo, c1_hi = float(m.group(1)), float(m.group(5)), float(m.group(6))
    assert abs(c1_obs - (sel["VD5"]["test"] - B["test"])) < 6e-4
    c1 = c1_lo > 0

    # ---------------------------------------------------------------- C2: paired bootstrap (paired_bootstrap.py CLI)
    cmd = [py, os.path.join(TOOLS, "paired_bootstrap.py"), "--system", f"VD5={sel['VD5']['path']}",
           "--system", f"V3B={T['V3B']['path']}", "--pairs", "VD5:V3B", "--n", "2000", "--seed", "0",
           "--expect", f"VD5={sel['VD5']['test']:.3f}", "--expect", "V3B=7.807"]
    print("\n=== C2: " + " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd))
    out2 = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    print(out2, end="")
    m = re.search(r"VD5 vs V3B: observed ([+-][\d.]+)\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]", out2)
    assert m, "could not parse the C2 line"
    c2_obs, c2_lo, c2_hi = map(float, m.groups())
    c2 = (c2_lo > 0 or c2_hi < 0) and c2_obs > 0

    print("\n=== DECISION (pre-registered rule)")
    print(f"  C1  VD5 - {base} = {c1_obs:+.3f}, max-T simultaneous 95% [{c1_lo:+.3f}, {c1_hi:+.3f}] "
          f"-> {'excludes 0: MET' if c1 else 'includes 0: NOT met'}")
    print(f"  C2  VD5 - V3B = {c2_obs:+.3f}, paired bootstrap 95% [{c2_lo:+.3f}, {c2_hi:+.3f}]; VD5 > V3B: "
          f"{c2_obs > 0} -> {'MET' if c2 else 'NOT met'}")
    if c1 and c2:
        claim = "C1 and C2: cross-modal ensembling beats the best single model on the video-disjoint split"
    elif c1:
        claim = "C1 only: ensembling beats the best single model on held-out videos; not attributable to modality"
    elif c2:
        claim = "C2 without C1: not one of the pre-registered claims; reported as a bounded estimate"
    else:
        claim = "neither: a bounded estimate (no claim)"
    print(f"  CLAIM: {claim}")

    # ---------------------------------------------------------------- secondary, descriptive
    train = {norm(t) for t in trt}
    print(f"\n=== SECONDARY (descriptive). unseen = vd_test reference never in vd_train text "
          f"(lowercased, whitespace-collapsed); content words exclude stitch_lexicon.STOP")
    R = {}
    for n, d in ((base, B), ("V3B", T["V3B"]), ("VD5", sel["VD5"])):
        r = readouts(d["hyps"], tt, STOP, train)
        assert abs(r["bleu4"] - d["test"]) < 1e-9, (n, r["bleu4"], d["test"])
        R[n] = r
        print(f"  {n:<9} BLEU {r['bleu4']:6.3f}  unseen({r['unseen_n']}) {r['bleu4_unseen']:6.3f}  "
              f"content P {r['content_precision']:6.2f}  R {r['content_recall']:6.2f}  len ratio {r['len_ratio']:.4f}  "
              f"WER {r['wer']:.4f}  distinct {r['distinct']}")
    for other in ("V3B", base):
        a, b = R["VD5"], R[other]
        print(f"  VD5 - {other:<9}: unseen BLEU {a['bleu4_unseen'] - b['bleu4_unseen']:+.3f}  content P "
              f"{a['content_precision'] - b['content_precision']:+.2f}  R {a['content_recall'] - b['content_recall']:+.2f}  "
              f"len ratio {a['len_ratio'] - b['len_ratio']:+.4f}  WER {a['wer'] - b['wer']:+.4f}")
    cmd = [py, os.path.join(TOOLS, "paired_bootstrap.py"), "--system", f"VD5={sel['VD5']['path']}",
           "--system", f"V3B={T['V3B']['path']}", "--system", f"{base}={B['path']}",
           "--pairs", f"VD5:V3B,VD5:{base}", "--train_manifest", os.path.join(MAN, "vd_train.tsv"),
           "--subset", "unseen", "--n", "2000", "--seed", "0"]
    print("\n  unseen-subset paired bootstrap (descriptive): "
          + " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd[1:]))
    out3 = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    print("".join("  " + l + "\n" for l in out3.splitlines()), end="")


if __name__ == "__main__":
    main()
