#!/bin/bash -l
# Subword/BPE target vocabulary — the output-side long-tail lever (never tried on the pose model).
# Word-level decoder emits UNK for rare words; BPE lets it compose them from pieces. SentencePiece
# support is already wired in Vocab; only a trained .model + the config flags are needed. Matched
# DWPose+CTC recipe (word-level baseline = 9.63 beam-5). SP models on durable $G so eval_beam finds them.
#SBATCH --job-name=bpe-subword
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer sentencepiece 2>/dev/null || true
DE=/scratch/pmc099/pprasad/dwpose; G=/group/pmc099/pprasad
cd "$G/code/phase2_baseline"
mkdir -p "$G/spm"
cut -f2- "$DE/manifests/train.tsv" > "$G/spm/ad_train_text.txt"
COM="--train_manifest $DE/manifests/train.tsv --val_manifest $DE/manifests/val.tsv --test_manifest $DE/manifests/test.tsv"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
REC="--batch_size 64 --epochs 40 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --seed 42 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
for V in 1000 2000; do
  echo "=== train SentencePiece BPE vocab_size=$V ==="
  python -c "import sentencepiece as spm; spm.SentencePieceTrainer.train(input='$G/spm/ad_train_text.txt', model_prefix='$G/spm/ad_bpe_$V', vocab_size=$V, model_type='bpe', character_coverage=1.0)"
  echo "=== DWPose+CTC with BPE-$V target vocab ==="
  python train_ctc.py $COM $POSE $REC \
    --text_tokenization sentencepiece --sentencepiece_model "$G/spm/ad_bpe_$V.model" \
    --out_dir "$DE/runs/ctc_bpe_$V"
  python tools/eval_beam.py --ckpt "$DE/runs/ctc_bpe_$V/best.pt" --manifest "$DE/manifests/test.tsv" \
    --beam 5 --device cuda || true
  cp -r "$DE/runs/ctc_bpe_$V" "$G/comm_dwpose_ctc_bpe_$V" 2>/dev/null || true
done
echo "BPE_SUBWORD_DONE"
