#!/bin/bash -l
# WAVE 5 step 1 -- full-dictionary constrained sentence generation, 4 shards (seeds 42..45), 125k each.
# Same generator, prompt and filters as the adopted pool (stitch_v3), plus --full_dictionary: the trie may use every
# MM-WLAuslan surface form, not only words already in the Auslan-Daily training vocabulary.
# Pre-registration: results/PREREG_WAVE5_SYNPRETRAIN_2026-09-22.md
#SBATCH --job-name=w5-gen
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge; module load cuda/12.6.3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/group/pmc099/pprasad/hf_cache
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; W5=$G/wave5; B=$G/scratch_backup; MM=$B/mmwl
cd "$W5/code" || { echo "FAIL: wave-5 code snapshot missing"; exit 1; }
md5sum tools/gen_stitch_sentences.py tools/constrained_gen.py tools/stitch_lexicon.py
T=$SLURM_ARRAY_TASK_ID; SEED=$((42 + T)); P=$W5/pool; mkdir -p "$P"
LAB="--labels $MM/labels/Train.json $MM/labels/Valid.json"
FIX="--constrained --full_dictionary --morphology --no_paren_alias"
AD="--ad_manifest $B/dwpose/manifests/train.tsv"

# gate <log> <out> <min_kept>: out_of_vocab must be 0 (the mask makes it impossible once the post-filter uses the same
# vocabulary); low_sign_coverage is a RATE (<= 1% of candidates), because coverage()'s greedy longest-first match can
# merge two adjacent trie phrases into one multi-word gloss (adopted pool: 38 of ~48,000). Those are filtered out anyway.
gate () {
  python - "$@" <<'PY'
import ast, os, re, sys
log, out, need = open(sys.argv[1]).read(), sys.argv[2], int(sys.argv[3])
assert "FULL DICTIONARY" in log, "the --full_dictionary branch did not run"
m = re.findall(r"rejection breakdown: (\{.*\})", log)
assert m, "no rejection breakdown in the log"
rej = ast.literal_eval(m[-1])
n = sum(1 for _ in open(out)) if os.path.exists(out) else 0
cand = n + sum(rej.values())
low = rej.get("low_sign_coverage", 0)
print(f"GATE: kept {n} | candidates {cand} | out_of_vocab {rej.get('out_of_vocab', 0)} | "
      f"low_sign_coverage {low} ({100 * low / max(cand, 1):.2f}%) | all {rej}")
if rej.get("out_of_vocab", 0):
    sys.exit("GATE FAIL: out_of_vocab rejections, so the post-filter and the trie disagree about the vocabulary")
if low > 0.01 * cand:
    sys.exit("GATE FAIL: low_sign_coverage above 1% of candidates")
if n < need:
    sys.exit(f"GATE FAIL: only {n} sentences kept (need {need})")
print("GATE_OK")
PY
}

echo "########## SMOKE (shard $T, seed $SEED) ##########"
python tools/gen_stitch_sentences.py $LAB $AD --out "$P/smoke_$T.txt" --device cuda --target 200 --max_rounds 4 \
  $FIX --batch_size 8 --per_prompt 16 --seed $SEED 2>&1 | tee "$P/smoke_$T.log"
gate "$P/smoke_$T.log" "$P/smoke_$T.txt" 50

# 4 x 125,000 = 500,000 raw leaves room for cross-shard duplicates before the 305,000 merge gate
# (adopted pool: ~41 kept per round of 96 sequences, ~2.7 s per round, duplicate share rising slowly).
echo "########## FULL (shard $T): target 125,000 ##########"
python tools/gen_stitch_sentences.py $LAB $AD --out "$P/shard_$T.txt" --device cuda \
  --target 125000 --max_rounds 12000 $FIX --batch_size 4 --per_prompt 24 --seed $SEED 2>&1 | tee "$P/shard_$T.log"
gate "$P/shard_$T.log" "$P/shard_$T.txt" 0
wc -l "$P/shard_$T.txt"
echo "W5_GEN_DONE shard $T"
