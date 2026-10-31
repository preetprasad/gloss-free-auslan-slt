#!/usr/bin/env python3
"""Make data/<arm>_s<seed>/beam5_{val,test}.json from the saved beam-5 decodes of the signer-disjoint experiment (wave 12g).
Run once in the private tree (it needs the saved decodes); make does not run it.
Each source file results/wave12g_138387/runs/<arm>_s<seed>/beam5_<split>_best.json is reduced to a SUBSET of its fields:
  beam_hyps              the decoded hypotheses, unchanged
  refs_sha1              SHA-1 of each reference sentence of the decode (the sentences themselves are not stored; check.py reads them
                         from the manifests rebuilt by `make data` and requires the digests to agree row for row)
  recorded_bleu4         beam_result.metrics[BLEU-4 key] of the decode, unchanged
  manifest_name          basename of the manifest the decode was made on
Dropped fields: checkpoint, manifest (cluster paths), split, beam, length_penalty, greedy, greedy_hyps, beam_result (other metrics), refs."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src" / "tools"), str(ROOT / "src")]
from arm_readouts import KEY                                  # noqa: E402

SRC = ROOT / "results" / "wave12g_138387" / "runs"
OUT = Path(__file__).resolve().parent / "data"
for arm in ("SD", "EXPa", "EXPb"):
    for seed in (42, 101, 202):
        for split in ("val", "test"):
            j = json.load(open(SRC / f"{arm}_s{seed}" / f"beam5_{split}_best.json"))
            name = f"w12g_{arm}_test.tsv" if split == "test" else "w12g_val.tsv"
            assert Path(j["manifest"]).name == name and j["beam"] == 5
            assert len(j["beam_hyps"]) == len(j["refs"])
            d = dict(manifest_name=name, beam_hyps=j["beam_hyps"],
                     refs_sha1=[hashlib.sha1(r.encode("utf-8")).hexdigest() for r in j["refs"]],
                     recorded_bleu4=j["beam_result"]["metrics"][KEY])
            p = OUT / f"{arm}_s{seed}" / f"beam5_{split}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            json.dump(d, open(p, "w"))
            print(p.relative_to(OUT), len(d["beam_hyps"]), d["recorded_bleu4"])
