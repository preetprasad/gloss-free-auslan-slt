#!/bin/bash -l
# D3 — REBUILD THE 1.66:1 SYNTHETIC CORPUS WITH ITS PROVENANCE SIDECAR (CPU only, no training).
#
# WHY. The adopted recipe (1.1:1 duration-stitch + pose augmentation) trains on stitch_scale2/corpus, which was built
# without --audit_jsonl, so the exact MM-WLAuslan gloss sequence rendered into each synthetic clip was never saved.
# Research round 1's top lever (rendered-sign CTC supervision of synthetic clips) needs that sequence, and its lexicon
# correctness arm needs the real share of letter glosses ('i' -> fingerspelled I, 'a' -> A) in the constrained pool.
# All earlier statistics used real train subtitles as a proxy.
#
# WHAT. Same builder, same flags as kaya/stitch_scale.sbatch (lines 93-102), plus --audit_jsonl, into a NEW directory.
# Then prove the rebuild is the corpus we trained on, clip by clip: array equality of KPS and BBOX for all 20,608 clips
# and identical sentence order. File md5 is recorded but not required (savez_compressed zip entries may carry
# timestamps). If any clip differs, the sidecar does NOT describe the trained corpus and the job fails loudly.
# Finally compute sidecar statistics on the 13,760-clip prefix (the 1.1:1 corpus) and on the full 20,608.
#SBATCH --job-name=stitch-audit
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; B=$G/scratch_backup
cd "$G/code/phase2_baseline"
OUT=$G/stitch_audit; mkdir -p "$OUT"
REF=$G/stitch_scale2/corpus
MM="$B/mmwl"
SENTS="$B/stitch_meta/stitch_v3/sentences_constrained.txt"

for d in train valid; do
  n=$(find "$MM/dwpose_clips/$d" -name "*.npz" | wc -l); echo "mmwl $d: $n npz"
  [ "$n" -gt 1000 ] || { echo "FAIL: MM-WLAuslan $d clips missing or too few ($n)"; exit 1; }
done
[ -f "$SENTS" ] || { echo "FAIL: constrained sentence pool missing"; exit 1; }
echo "sentence pool: $(wc -l < "$SENTS") lines"
NREF=$(find "$REF/clips" -name "*.npz" | wc -l); echo "reference corpus clips: $NREF"
[ "$NREF" -eq 20608 ] || { echo "FAIL: reference corpus has $NREF clips, expected 20608"; exit 1; }

COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json \
  --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
  --sentences $SENTS --min_cov 0.9 --min_signs 3 --realizations 1 --seed 42 \
  --morphology --no_paren_alias --sign_dur"
echo "=== rebuild with sidecar ==="
python tools/build_posestitch_corpus_v2.py $COMMON --out_dir "$OUT/corpus" --max_clips 20608 \
  --audit_jsonl "$OUT/corpus/audit.jsonl"

echo "=== identity check against the trained corpus + sidecar statistics ==="
python - "$OUT/corpus" "$REF" <<'PYCHK'
import hashlib, json, os, sys
from collections import Counter
import numpy as np
new, ref = sys.argv[1], sys.argv[2]
rows = lambda p: [l.rstrip("\n").split("\t", 1) for l in open(p, encoding="utf-8") if "\t" in l]
nm, rm = rows(os.path.join(new, "posestitch_v2_train.tsv")), rows(os.path.join(ref, "posestitch_v2_train.tsv"))
assert len(nm) == len(rm) == 20608, (len(nm), len(rm))
text_same = sum(a[1] == b[1] for a, b in zip(nm, rm))
name_same = sum(os.path.basename(a[0]) == os.path.basename(b[0]) for a, b in zip(nm, rm))
arr_diff, md5_same, first_bad = 0, 0, None
for i, (a, b) in enumerate(zip(nm, rm)):
    x, y = np.load(a[0]), np.load(b[0])
    same = np.array_equal(x["KPS"], y["KPS"]) and np.array_equal(x["BBOX"], y["BBOX"])
    if not same:
        arr_diff += 1; first_bad = first_bad if first_bad is not None else i
    md5_same += hashlib.md5(open(a[0], "rb").read()).digest() == hashlib.md5(open(b[0], "rb").read()).digest()
print(f"manifest rows 20608 | sentence text identical {text_same}/20608 | clip names identical {name_same}/20608")
print(f"KPS+BBOX arrays identical {20608 - arr_diff}/20608 (first difference at row {first_bad}) | file md5 identical {md5_same}/20608")
aud = [json.loads(l) for l in open(os.path.join(new, "audit.jsonl"), encoding="utf-8")]
assert len(aud) == 20608 and all(os.path.basename(r["clip_path"]) == os.path.basename(n[0]) for r, n in zip(aud, nm))
if arr_diff or text_same != 20608 or name_same != 20608:
    print("FAIL: the rebuild does NOT reproduce the trained corpus; the sidecar must not be used for it"); sys.exit(1)
print("IDENTICAL: the sidecar describes the trained corpus exactly")
def stats(recs, tag):
    segs = [s for r in recs for s in r["segments"]]
    words = sum(len(r["text"].split()) for r in recs)
    g = Counter(s["gloss"] for s in segs)
    letters = {k: v for k, v in g.items() if len(k) == 1 and k.isalpha()}
    print(f"[{tag}] clips {len(recs)} | segments {len(segs)} | signs/sentence {len(segs)/len(recs):.2f} | "
          f"whitespace tokens/sentence {words/len(recs):.2f} | signs per token {len(segs)/words:.3f} | "
          f"distinct glosses {len(g)}")
    print(f"[{tag}] single-letter glosses: {sum(letters.values())} segments = {100*sum(letters.values())/len(segs):.2f}% "
          f"| top letters {sorted(letters.items(), key=lambda kv: -kv[1])[:8]}")
    print(f"[{tag}] clips containing gloss 'I': {sum(any(s['gloss']=='I' for s in r['segments']) for r in recs)} "
          f"| containing 'A': {sum(any(s['gloss']=='A' for s in r['segments']) for r in recs)}")
    print(f"[{tag}] top glosses {g.most_common(25)}")
    out = [s["output_frames"] for s in segs]
    print(f"[{tag}] sign output frames mean {np.mean(out):.1f} median {np.median(out):.0f}")
stats(aud[:13760], "1.1:1 prefix (trained with pose-aug)")
stats(aud, "full 1.66:1")
PYCHK
echo "STITCH_AUDIT_DONE"
