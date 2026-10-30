#!/bin/bash -l
# Multi-reference paraphrase augmentation (HANDOFF next-action #2).
#
# Auslan-Daily gives ONE English reference per clip, so cross-entropy penalises every other
# valid wording — a single-reference / exposure-bias ceiling that no pose-side lever can touch.
# Stage A: Qwen2.5-3B paraphrases each unique TRAIN subtitle (train manifest only — val/test
# are never touched). Stage B: DWPose+CTC trains with a per-epoch chance of using a paraphrase
# instead of the reference. Paraphrases are filtered to the existing vocabulary, so the output
# space is identical to the control and the comparison stays clean.
# Arms: in-batch control (prob 0) + prob 0.3 + prob 0.5.
#SBATCH --job-name=para-aug
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q "transformers==4.46.3" "accelerate==1.1.1" sentencepiece jiwer 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
PARA="$DE/runs/paraphrases_train.tsv"

echo "=== SMOKE GATE: paraphrase generation on 32 sentences ==="
python tools/gen_paraphrases.py --manifest "$DE/manifests/train.tsv" \
  --out "$DE/runs/paraphrases_smoke.tsv" --device cuda --limit 32 --num_paraphrases 3
test -s "$DE/runs/paraphrases_smoke.tsv" || { echo "SMOKE FAIL: no paraphrases"; exit 1; }
head -6 "$DE/runs/paraphrases_smoke.tsv"

echo "=== STAGE A: paraphrase every unique training subtitle ==="
python tools/gen_paraphrases.py --manifest "$DE/manifests/train.tsv" \
  --out "$PARA" --device cuda --num_paraphrases 3 --batch_size 32
wc -l "$PARA"

echo "=== VOCAB-RETENTION CHECK (how much augmentation actually survives filtering) ==="
python - <<PY
import sys; sys.path.insert(0, ".")
from config import Config
from slt.data import Vocab, tokenize
cfg = Config()
train_txt = [l.rstrip("\n").split("\t")[1] for l in open("$DE/manifests/train.tsv") if l.strip()]
vocab = Vocab.build(train_txt, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
known = set(vocab.itos)
kept = total = 0
sents = set()
for ln in open("$PARA", encoding="utf-8"):
    if "\t" not in ln: continue
    k, v = ln.rstrip("\n").split("\t", 1)
    total += 1
    if all(t in known for t in tokenize(v.strip(), cfg.text_tokenization)):
        kept += 1; sents.add(" ".join(k.lower().split()))
print(f"[retention] {kept}/{total} paraphrases in-vocab ({100.0*kept/max(1,total):.1f}%), "
      f"covering {len(sents)} of {len(set(' '.join(t.lower().split()) for t in train_txt))} sentences")
PY

COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"

echo "=== ARM 0: in-batch CONTROL (no paraphrase augmentation) ==="
python train_ctc.py $COM $POSE $REC --out_dir "$DE/runs/para_control"
python tools/eval_beam.py --ckpt "$DE/runs/para_control/best.pt" \
  --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
cp -r "$DE/runs/para_control" "$G/comm_dwpose_para_control" 2>/dev/null || true

for P in 0.3 0.5; do
  echo "=== ARM: paraphrase_prob=$P ==="
  python train_ctc.py $COM $POSE $REC --paraphrase_map "$PARA" --paraphrase_prob $P \
    --out_dir "$DE/runs/para_p$P"
  python tools/eval_beam.py --ckpt "$DE/runs/para_p$P/best.pt" \
    --manifest "$DE/manifests/test.tsv" --beam 5 --device cuda || true
  cp -r "$DE/runs/para_p$P" "$G/comm_dwpose_para_p$P" 2>/dev/null || true
done

echo "=== SUMMARY (compare arms to the in-batch control above) ==="
for R in para_control para_p0.3 para_p0.5; do
  python -c "
import json
p='$DE/runs/$R/progress.json'
try:
    d=json.load(open(p)); t=d.get('test',{})
    print('$R', 'val_best=%.3f@ep%d' % (d['best']['bleu4'], d['best']['epoch']),
          'test_bleu4=%.3f' % t.get('bleu4', float('nan')), 'wer=%.3f' % t.get('wer', float('nan')))
except Exception as e:
    print('$R', 'NO RESULT', e)
"
done
echo "PARAPHRASE_AUG_DONE"
