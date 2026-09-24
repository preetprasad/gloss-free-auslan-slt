#!/bin/bash -l
# NEWS five-seed nSA ensemble (rank 4, design round 2026-09-24). Decode only, CPU. Rules fixed in
# results/PREREG_WAVE4_2026-09-22.md, "Addendum, 24 Sep 08:24 AWST", written before any Stage-2 nSA number existed.
# Phase 1: each member alone + the 5-member ensemble on news_val (LP sweep).  Gate in code: futility if
# ensemble val <= best single val + 0.10 -> stop, NO test decode.  Phase 2: test decodes; the analysis reads each
# system at its own validation-argmax LP.  Code: the local-tools snapshot $G/seedens/code (preprocessing guard).
#SBATCH --job-name=news-ens
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
G=/group/pmc099/pprasad; M=$G/scratch_backup/dwpose/manifests; RN=$G/wave4/runs_news; OUT=$G/news_ens; mkdir -p "$OUT"
cd "$G/seedens/code" || { echo "FAIL: snapshot missing"; exit 1; }
echo "node $(hostname)"; md5sum tools/ensemble_mbr_wer.py slt/preproc_guard.py ensemble_decode.py
CKS=(); for s in 42 101 202 303 404; do f=$RN/nSA_s$s/best.pt; [ -s "$f" ] || { echo "FAIL: missing $f"; exit 1; }; CKS+=("$f"); done
python - "$M/news_val.tsv" "$M/news_test.tsv" <<'PY'
import sys
v = {l.split("\t")[0] for l in open(sys.argv[1]) if "\t" in l}; t = {l.split("\t")[0] for l in open(sys.argv[2]) if "\t" in l}
assert v and t and not v & t, "news val/test overlap or empty"; print(f"news_val {len(v)} / news_test {len(t)} / overlap 0")
PY
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame --max_src_frames 512"
dec () {   # dec <system> <ckpts> <manifest> <split> <lps>
  local d="$OUT/$1_$4"
  python tools/ensemble_mbr_wer.py --ckpts "$2" --eval_manifest "$3" $POSE --beam 10 --lps "$5" --device cpu \
    --dump_dir "$d" > "$d.log" 2>&1 || { echo "FAIL: decode $1 $4"; tail -20 "$d.log"; exit 1; }
  echo "done $1 $4"
}
LPS=0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,2.0
ENS=$(IFS=,; echo "${CKS[*]}")
for i in 0 1 2 3 4; do dec "s$i" "${CKS[$i]}" "$M/news_val.tsv" val "$LPS"; done
dec ens5 "$ENS" "$M/news_val.tsv" val "$LPS"
pick () { python - "$@" <<'PY'
import re, sys
best = None
for l in open(sys.argv[1]):
    m = re.match(r"\s+LP=([0-9.]+)\s+BLEU-4=\s*([0-9.]+)", l)
    if m and (best is None or float(m.group(2)) > best[1]): best = (m.group(1), float(m.group(2)))
print(best[0], best[1])
PY
}
read ELP EVAL < <(pick "$OUT/ens5_val.log")
if [ "$ELP" = 2.0 ]; then   # prereg: extend the grid on validation only
  dec ens5x "$ENS" "$M/news_val.tsv" val 2.0,2.4,2.8,3.2
  read XLP XVAL < <(pick "$OUT/ens5x_val.log")
  if python -c "import sys; sys.exit(0 if $XVAL > $EVAL else 1)"; then ELP=$XLP; EVAL=$XVAL; fi
fi
BESTS=""; BVAL=-1
for i in 0 1 2 3 4; do read l v < <(pick "$OUT/s${i}_val.log"); echo "s$i val $v @LP$l"
  if python -c "import sys; sys.exit(0 if $v > $BVAL else 1)"; then BVAL=$v; BESTS=$i; fi; done
echo "ENSEMBLE val $EVAL @LP$ELP | best single s$BESTS val $BVAL" | tee "$OUT/phase1.txt"
if python -c "import sys; sys.exit(0 if $EVAL <= $BVAL + 0.10 else 1)"; then
  echo "FUTILITY: ensemble val <= best single + 0.10 -> ensembling does not transfer to News on validation; no test decode" | tee -a "$OUT/phase1.txt"
  exit 0
fi
echo "futility passed -> test decodes" | tee -a "$OUT/phase1.txt"
for i in 0 1 2 3 4; do dec "s$i" "${CKS[$i]}" "$M/news_test.tsv" test "$LPS"; done
TL=$LPS   # the ensemble test sweep must contain its validation LP (it may come from the extended grid)
case ",$LPS," in *",$ELP,"*) ;; *) TL="$LPS,$ELP" ;; esac
dec ens5 "$ENS" "$M/news_test.tsv" test "$TL"
touch "$OUT/DONE"; echo "NEWS_ENS_DONE"
