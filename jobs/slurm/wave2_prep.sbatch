#!/bin/bash -l
# WAVE 2 PREP — build the PAIRED synthetic corpora for the fingerspelled-letter fix (R2-E1).
# Pre-registration: results/PREREG_WAVE2_R2_2026-09-22.md (written before any arm was trained).
#
# The defect (measured by the stitch audit, job 1216905): MM-WLAuslan ships the fingerspelling alphabet as 24
# single-character glosses, so the pronoun "I" is stitched as the LETTER I and the article "a" as the LETTER A.
# 8,187 of 94,428 segments (8.67%) of the corpus the adopted recipe trains on are single letters.
#
# The pairing rule that makes this interpretable: letter_fix drops ~4% of sentences below cov>=0.9, so an ordinary
# rebuild would draw further down the pool and differ in WHICH SENTENCES it contains as well as which sign is shown
# -- the exact confound that cost job 1193265 its interpretability. Both corpora are therefore built from the SAME
# pre-filtered sentence list (passing min_cov 0.9 AND min_signs 3 under BOTH lexicons), same seed, same budget, and
# the job FAILS unless the two manifests are identical in length, row order and English text.
#SBATCH --job-name=wave2-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
MM=$G/scratch_backup/mmwl
POOL=$G/scratch_backup/stitch_meta/stitch_v3/sentences_constrained.txt
SRC=$G/stitch_scale2
OUT=$G/wave2_lf; mkdir -p "$OUT"
md5sum tools/stitch_lexicon.py tools/build_posestitch_corpus_v2.py

LAB="$MM/labels/Train.json $MM/labels/Valid.json"
NPZ="$MM/dwpose_clips/train $MM/dwpose_clips/valid"
echo "pool: $(wc -l < "$POOL") sentences | mmwl train npz $(find $MM/dwpose_clips/train -name '*.npz' | wc -l) valid $(find $MM/dwpose_clips/valid -name '*.npz' | wc -l)"

# --- 1. the shared sentence list: accepted by BOTH lexicons, so composition cannot differ between arms ---
python - "$POOL" "$OUT/sentences_paired.txt" $LAB <<'PYFILT'
import sys
sys.path.insert(0, "tools")
from stitch_lexicon import build_lexicon, coverage, content_words
pool_p, out_p = sys.argv[1], sys.argv[2]
labels = sys.argv[3:]
off = build_lexicon(labels, morphology=True, paren_alias=False)
on  = build_lexicon(labels, morphology=True, paren_alias=False, letter_fix=True)
# the OFF lexicon must reproduce the one the adopted corpus was built with
assert len(off.phrase2gloss) == 3090, f"OFF lexicon has {len(off.phrase2gloss)} surface forms, expected 3090"
dropped = sorted(set(off.phrase2gloss) - set(on.phrase2gloss))
assert len(dropped) == 23 and all(len(x) == 1 for x in dropped), dropped
assert off.phrase2gloss["i"] == "i" and on.phrase2gloss["i"] == "me"
print(f"lexicons OK: OFF {len(off.phrase2gloss)} forms, ON {len(on.phrase2gloss)}; dropped {''.join(dropped)}; i -> me")

sents = [l.strip() for l in open(pool_p, encoding="utf-8") if l.strip()]
kept, n_off, n_on = [], 0, 0
for t in sents:
    if not content_words(t):
        continue
    c_off, s_off = coverage(t, off)
    c_on,  s_on  = coverage(t, on)
    ok_off = c_off >= 0.9 and len(s_off) >= 3
    ok_on  = c_on  >= 0.9 and len(s_on)  >= 3
    n_off += ok_off; n_on += ok_on
    if ok_off and ok_on:
        kept.append(t)
print(f"pool {len(sents)} | accepted by OFF {n_off} | by ON {n_on} | by BOTH {len(kept)}")
assert len(kept) >= 14500, f"only {len(kept)} paired sentences, need >= 14500 to reach 13760 clips"
open(out_p, "w", encoding="utf-8").write("\n".join(kept) + "\n")
PYFILT

COMMON="--labels $LAB --npz_dirs $NPZ --sentences $OUT/sentences_paired.txt \
  --min_cov 0.9 --min_signs 3 --realizations 1 --seed 42 --morphology --no_paren_alias --sign_dur --max_clips 13760"

# --- 2. the two corpora: identical flags, identical pool, identical seed; only --letter_fix moves ---
echo "=== corpus A (letter_fix OFF = the adopted recipe's lexicon) ==="
python tools/build_posestitch_corpus_v2.py $COMMON --out_dir "$OUT/corpus_a"
echo "=== corpus LF (letter_fix ON) ==="
python tools/build_posestitch_corpus_v2.py $COMMON --out_dir "$OUT/corpus_lf" --letter_fix

A="$OUT/corpus_a/posestitch_v2_train.tsv"; LF="$OUT/corpus_lf/posestitch_v2_train.tsv"
for f in "$A" "$LF"; do n=$(wc -l < "$f"); echo "$(basename $(dirname $f)): $n clips"
  [ "$n" -eq 13760 ] || { echo "FAIL: $f has $n clips, expected 13760"; exit 1; }; done

# --- 3. the pairing assertion: same sentences, same order, only the signs differ ---
python - "$A" "$LF" <<'PYPAIR'
import sys
a = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8")]
b = [l.rstrip("\n").split("\t") for l in open(sys.argv[2], encoding="utf-8")]
assert len(a) == len(b) == 13760, (len(a), len(b))
same_text = sum(x[1] == y[1] for x, y in zip(a, b))
same_name = sum(x[0].rsplit("/", 1)[-1] == y[0].rsplit("/", 1)[-1] for x, y in zip(a, b))
print(f"rows {len(a)} | identical English text {same_text}/{len(a)} | identical clip names {same_name}/{len(a)}")
if same_text != len(a):
    print("first mismatches:", [(i, x[1], y[1]) for i, (x, y) in enumerate(zip(a, b)) if x[1] != y[1]][:3])
sys.exit(0 if same_text == len(a) and same_name == len(a) else 1)
PYPAIR

# --- 4. joint manifests against the same 12,439 real clips the adopted recipe uses ---
REAL="$SRC/manifests/train.tsv"; echo "real train: $(wc -l < "$REAL") clips"
cat "$REAL" "$A"  > "$OUT/joint_a.tsv"
cat "$REAL" "$LF" > "$OUT/joint_lf.tsv"
for f in "$OUT/joint_a.tsv" "$OUT/joint_lf.tsv"; do
  n=$(wc -l < "$f"); echo "$(basename $f): $n rows"; [ "$n" -eq 26199 ] || { echo "FAIL: $f has $n rows"; exit 1; }; done

# --- 5. every referenced clip must exist, and the two arms' letter statistics must differ as designed ---
python - "$OUT/joint_a.tsv" "$OUT/joint_lf.tsv" "$OUT/corpus_a/audit.json" <<'PYCHK'
import json, os, sys, collections
for p in sys.argv[1:3]:
    rows = [l.split("\t", 1)[0] for l in open(p) if "\t" in l]
    missing = sum(not os.path.exists(c) for c in rows)
    print(f"{os.path.basename(p)}: {len(rows)} clips, {missing} missing")
    assert missing == 0, p
PYCHK
echo "WAVE2_PREP_DONE"
