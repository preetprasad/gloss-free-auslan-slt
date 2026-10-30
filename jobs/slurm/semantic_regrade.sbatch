#!/bin/bash -l
# t3-errs6 Part 2 — semantic re-grade with the 12-Aug judge, re-downloaded to a DURABLE cache (approved by Preet
# 2026-09-22). Pre-registered: results/PREREG_WAVE4_2026-09-22.md §7. HARD GATE FIRST: RGB-fusion must reproduce
# mean 1.259 / 9.9% / 37.0%, or nothing else is graded.
#SBATCH --job-name=sem-regrade
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge; module load Anaconda3/2024.06
source /group/pmc099/pprasad/envs/auslan/bin/activate
export OMP_NUM_THREADS=16 HF_HOME=/group/pmc099/pprasad/hf_cache
G=/group/pmc099/pprasad; S=$G/code/results/semantic_grading; OUT=$G/semantic_regrade; mkdir -p "$OUT" "$HF_HOME"
MODEL=Qwen/Qwen2.5-3B-Instruct
md5sum "$S/grade_semantic_hf.py"; python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
grade () { python "$S/grade_semantic_hf.py" --in "$1" --field hyp_beam --model "$MODEL" --device cpu --out "$OUT/$2.graded_hf.csv" 2>&1 | grep -v "^\[grade\]" | tee "$OUT/$2.log"; }
echo "=== GATE: RGB-fusion (recorded mean 1.259 / 9.9% / 37.0%) ==="
grade "$G/semantic_hyps/hyps_rgbfusion.jsonl" rgbfusion
python - "$OUT/rgbfusion.log" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r"mean=([\d.]+) \| %adequate\(>=3\)=([\d.]+)% \| %partial\+\(>=2\)=([\d.]+)%", t)
mean, ad, pp = map(float, m.groups())
ok = abs(mean - 1.259) <= 0.005 and abs(ad - 9.9) <= 0.1 and abs(pp - 37.0) <= 0.1
print(f"GATE {'PASSED' if ok else 'FAILED'}: mean {mean} adequate {ad}% partial+ {pp}% vs 1.259 / 9.9 / 37.0")
sys.exit(0 if ok else 1)
PY
for f in dwpose_ctc vlp; do echo "=== $f ==="; grade "$G/semantic_hyps/hyps_$f.jsonl" "$f"; done
echo "=== S6 (beam-10 / LP 0.3) ==="; grade "$G/ladder_verify/dumps_s6/hyps_LP=0.3.jsonl" s6
echo "SEM_REGRADE_DONE"
