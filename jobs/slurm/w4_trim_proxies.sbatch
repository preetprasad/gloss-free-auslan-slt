#!/bin/bash -l
# w4-trim proxies owed by PREREG_WAVE4 §1 ("also report the proxies with the unfavourable ones included: join-gap
# ratio, velocity median, pause density, and separability with its margin over chance"). CPU only, no model.
# Corpus A (adopted, --trim_frac 0.15) vs T (--trim_frac 0.0), identical seed / settings / real train manifest.
#SBATCH --job-name=w4-trim-prox
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; C=$G/wave4/corpora; OUT=$G/wave4/trim_proxies; mkdir -p "$OUT"
cd "$G/seedens/code"; md5sum tools/audit_posestitch_motion.py
for arm in A T; do
  python tools/audit_posestitch_motion.py --audit_jsonl "$C/audit_$arm.jsonl" \
    --real_train_manifest "$G/scratch_backup/dwpose/manifests/train.tsv" --out_json "$OUT/proxies_$arm.json" \
    --seed 42 --max_pairs 1000 --coord_norm auto > "$OUT/proxies_$arm.log" 2>&1 || { echo "FAIL $arm"; tail -20 "$OUT/proxies_$arm.log"; exit 1; }
  echo "== $arm"; tail -25 "$OUT/proxies_$arm.log"
done
echo TRIM_PROXIES_DONE
