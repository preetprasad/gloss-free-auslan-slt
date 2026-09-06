#!/bin/bash -l
# CONSTRAINED GENERATION — remove the sentence bottleneck, and fix the corpus's known defects.
#
# Sentence generation is FILTER-bound, not sampler-bound: real human Auslan-Daily sentences pass
# our own filters only 9.34% of the time, and the LLM already runs at 6.75% = 72% of that natural
# -text ceiling, so prompting is spent. Conditionally, out_of_vocab removes 50.4% of survivors and
# low_sign_coverage then removes 85.5% of what remains.
#
# `tools/constrained_gen.py` masks the logits to a signable-phrase trie, which makes BOTH filters
# structurally impossible -- coverage is computed over content words, and every non-STOP phrase
# the mask admits is signable by construction, so content coverage is identically 1.0. Expected
# ~93% acceptance = ~13.8x yield. The state machine is unit-tested (5,000 randomised generations,
# zero violations; prefix-closure and terminal-and-prefix branches verified).
#
# This run ALSO applies the two lexicon fixes measured on 2026-09-05:
#   --morphology      inflected words match citation-form glosses (Auslan does not inflect for
#                     English plural/tense). Generator vocab 1,134 -> 1,565, coverage 51.1 -> 56.8%.
#   --no_paren_alias  stop treating a gloss's bracket as a synonym. It is usually a SENSE
#                     DISAMBIGUATOR, and aliasing it stitched wrong-sense signs under correct
#                     English words -- 4.8% of all matched sign tokens ("at"->PERSON (AT) 326x,
#                     "watch"->WIND UP (WATCH) 233x, "water"->WAVE (WATER) 141x).
#
# GENERATION ONLY -- no training. The post-hoc filters stay on as ASSERTIONS: under constraint they
# must reject ~0 on out_of_vocab/low_sign_coverage, and any rejection there means a bug in the mask.
#SBATCH --job-name=gen-constr
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q "transformers==4.46.3" "accelerate==1.1.1" sentencepiece 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
MM=/scratch/pmc099/pprasad/mmwl; ST=/scratch/pmc099/pprasad/stitch_v3
cd "$G/code/phase2_baseline"; mkdir -p "$ST"
LAB="--labels $MM/labels/Train.json $MM/labels/Valid.json"
FIX="--morphology --no_paren_alias"

echo "########## SMOKE GATE — constrained decoding must reject ~0 on vocab/coverage ##########"
python tools/gen_stitch_sentences.py $LAB --ad_manifest "$DE/manifests/train.tsv" \
  --out "$ST/smoke.txt" --device cuda --target 200 --max_rounds 4 \
  --constrained $FIX --batch_size 8 --per_prompt 16 2>&1 | tee "$ST/smoke.log"
python - <<'PY'
import os, re, sys
ST = "/scratch/pmc099/pprasad/stitch_v3"
log = open(f"{ST}/smoke.log").read()
m = re.findall(r"rejection breakdown: (\{.*\})", log)
rej = eval(m[-1]) if m else {}
bad = {k: v for k, v in rej.items() if k in ("out_of_vocab", "low_sign_coverage")}
n = sum(1 for _ in open(f"{ST}/smoke.txt")) if os.path.exists(f"{ST}/smoke.txt") else 0
print(f"SMOKE: kept {n} | structural rejections {bad or 'NONE'} | other {rej}")
if bad:
    sys.exit(f"SMOKE FAIL: the mask let through {bad} -- constrained decoding has a bug")
if n < 50:
    sys.exit(f"SMOKE FAIL: only {n} sentences from 4 rounds; yield far below the ~93% expected")
print("SMOKE_OK")
PY
head -8 "$ST/smoke.txt"

echo "########## FULL RUN — target 50,000 unique sentences ##########"
python tools/gen_stitch_sentences.py $LAB --ad_manifest "$DE/manifests/train.tsv" \
  --out "$ST/sentences_constrained.txt" --device cuda \
  --target 50000 --max_rounds 500 --constrained $FIX \
  --batch_size 4 --per_prompt 24 --seed 7
wc -l "$ST/sentences_constrained.txt"

echo "########## CORPUS STATS ##########"
python - <<'PY'
import sys; sys.path.insert(0, ".")
from tools.stitch_lexicon import build_lexicon, coverage, content_words
L = "/scratch/pmc099/pprasad/mmwl/labels"; ST = "/scratch/pmc099/pprasad/stitch_v3"
lex = build_lexicon([f"{L}/Train.json", f"{L}/Valid.json"], morphology=True, paren_alias=False)
sents = [l.strip() for l in open(f"{ST}/sentences_constrained.txt") if l.strip()]
cov = [coverage(s, lex)[0] for s in sents[:4000]]
wl = [len(s.split()) for s in sents]
print(f"sentences: {len(sents)} unique | mean words {sum(wl)/len(wl):.1f} "
      f"(min {min(wl)} max {max(wl)}) | mean content coverage {100*sum(cov)/len(cov):.1f}% "
      f"| fraction at 100% coverage {100*sum(c >= 0.999 for c in cov)/len(cov):.1f}%")
vocab = set()
for s in sents: vocab.update(s.split())
print(f"distinct words used: {len(vocab)}")
PY
echo "GEN_CONSTRAINED_DONE"
