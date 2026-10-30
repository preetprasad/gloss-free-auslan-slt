#!/bin/bash -l
# WAVE 5 step 2 (CPU) -- merge the 4 generator shards, build 305,000 synthetic clips in 10 contiguous shards with the
# ADOPTED builder flags, gate them, write the nested prefix manifests (25k / 100k / 200k / 300k) and ONE fixed
# vocabulary for every arm. The training array is afterok on this job.
# Pre-registration: results/PREREG_WAVE5_SYNPRETRAIN_2026-09-22.md sections 3 and 7
#SBATCH --job-name=w5-prep
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; W5=$G/wave5; W5S=/scratch/pmc099/pprasad/wave5; B=$G/scratch_backup; MM=$B/mmwl; M=$B/dwpose/manifests; P=$W5/pool
cd "$W5/code" || { echo "FAIL: wave-5 code snapshot missing"; exit 1; }
md5sum tools/merge_sentence_pool.py tools/w5_prep_check.py tools/build_vocab_file.py tools/build_posestitch_corpus_v2.py
python tools/merge_sentence_pool.py selftest
python tools/w5_prep_check.py selftest

echo "########## MERGE ##########"
python tools/merge_sentence_pool.py --shards "$P"/shard_0.txt "$P"/shard_1.txt "$P"/shard_2.txt "$P"/shard_3.txt \
  --exclude "$M/train.tsv" "$M/val.tsv" "$M/test.tsv" --out "$P/pool.txt" --min_size 305000 --seed 42

echo "########## BUILD: 10 x 30,500 sentences, contiguous ranges, seed 42 + k ##########"
# The ~305,000 clip files go to /scratch: the /group fileset had 1.6M free inodes of 247M (22 Sep), /scratch 26M.
# The corpus is regenerable (pool.txt stays on /group; the builder is deterministic, wave-4 prep rebuilt its corpus
# array-for-array) and every epoch reads every clip, so the access-time purge cannot touch it while the wave runs.
df -hi /group/pmc099 /scratch/pmc099 | sed 's/^/  inodes: /'
C=$W5S/corpora; rm -rf "$C"; mkdir -p "$C"
head -n 305000 "$P/pool.txt" | split -l 30500 -d -a 2 - "$C/sent_"
COMMON="--labels $MM/labels/Train.json $MM/labels/Valid.json \
  --npz_dirs $MM/dwpose_clips/train $MM/dwpose_clips/valid \
  --min_cov 0.9 --min_signs 3 --morphology --no_paren_alias --sign_dur --realizations 1"
for k in 00 01 02 03 04 05 06 07 08 09; do
  mkdir -p "$C/s$k"
  python tools/build_posestitch_corpus_v2.py $COMMON --sentences "$C/sent_$k" --seed $((42 + 10#$k)) \
    --out_dir "$C/s$k" --audit_jsonl "$C/s$k/audit.jsonl" > "$C/s$k.log" 2>&1 &
done
fail=0; for j in $(jobs -p); do wait "$j" || fail=1; done
grep -h "wrote" "$C"/s*.log
[ "$fail" = 0 ] || { echo "FAIL: a corpus shard exited non-zero"; exit 1; }

echo "########## GATE + PREFIX MANIFESTS ##########"
MW=$W5/manifests; rm -rf "$MW"; mkdir -p "$MW"
python tools/w5_prep_check.py --shard_dirs "$C"/s00 "$C"/s01 "$C"/s02 "$C"/s03 "$C"/s04 "$C"/s05 "$C"/s06 "$C"/s07 \
  "$C"/s08 "$C"/s09 --refs "$M/train.tsv" "$M/val.tsv" "$M/test.tsv" --ad_train "$M/train.tsv" --out_dir "$MW" \
  --sizes 25000 100000 200000 300000 --sentences_in 305000

echo "########## FIXED VOCABULARY ##########"
python tools/build_vocab_file.py --out "$MW/vocab.json" \
  --manifests "$M/train.tsv" "$G/stitch_scale2/joint_13760.tsv" "$MW/syn_300000.tsv"
python - "$MW/vocab.json" "$G/stitch_scale2/joint_13760.tsv" "$MW"/syn_25000.tsv "$MW"/syn_100000.tsv \
  "$MW"/syn_200000.tsv "$MW"/syn_300000.tsv <<'PY'
import json, sys
sys.path.insert(0, ".")
from config import Config
from slt.data import tokenize
cfg, itos = Config(), json.load(open(sys.argv[1]))
stoi = set(itos)
for m in sys.argv[2:]:
    oov = sum(w not in stoi for l in open(m) if "\t" in l for w in tokenize(l.rstrip("\n").split("\t")[1],
                                                                           cfg.text_tokenization))
    print(f"G-vocab {m.split('/')[-1]}: {oov} tokens outside the fixed vocabulary")
    assert oov == 0
print(f"G-vocab OK ({len(itos)} entries)")
PY
echo "W5_PREP_OK"
