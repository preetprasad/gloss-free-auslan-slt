#!/usr/bin/env bash
# Closest public GASLT recipe reconstruction for Auslan Communication.
# Faithful downstream settings: English BPEmb-25k/300-D, 2+2 layers, N=7,
# dropout .5, similarity transfer, smoothing .4, batch 32 and paper optimiser.
# Remaining non-public variable: Auslan pose points -> visual feature adapter.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
run_dir="${1:-${project_dir}/comm_gaslt_paper_recipe_full}"
train_manifest="${AUSLAN_TRAIN_MANIFEST:-${project_dir}/comm_run/manifests/train.tsv}"
val_manifest="${AUSLAN_VAL_MANIFEST:-${project_dir}/comm_run/manifests/val.tsv}"
test_manifest="${AUSLAN_TEST_MANIFEST:-${project_dir}/comm_run/manifests/test.tsv}"
auslan_epochs="${AUSLAN_EPOCHS:-60}"
auslan_batch_size="${AUSLAN_BATCH_SIZE:-32}"
auslan_grad_accum_steps="${AUSLAN_GRAD_ACCUM_STEPS:-1}"
auslan_resume_checkpoint="${AUSLAN_RESUME_CHECKPOINT:-}"
bpemb_dir="${project_dir}/assets/bpemb/en"
sp_model="${bpemb_dir}/en.wiki.bpe.vs25000.model"
word_vectors="${bpemb_dir}/en.wiki.bpe.vs25000.d300.w2v.bin"

mkdir -p "${run_dir}/out"
resume_args=()
if [[ -n "${auslan_resume_checkpoint}" ]]; then
  resume_args=(--resume_checkpoint "${auslan_resume_checkpoint}")
fi

python -u "${script_dir}/train.py" \
  --model_arch gaslt \
  --train_manifest "${train_manifest}" \
  --val_manifest "${val_manifest}" \
  --test_manifest "${test_manifest}" \
  --keypoint_subset body_hands \
  --use_confidence true \
  --coord_norm bbox \
  --bbox_format xywh \
  --conf_threshold 0.1 \
  --drop_zero_frames true \
  --max_src_frames 400 \
  --max_tgt_tokens 80 \
  --decode_max_len 40 \
  --text_tokenization sentencepiece \
  --sentencepiece_model "${sp_model}" \
  --pretrained_target_embeddings "${word_vectors}" \
  --visual_feature_dim 0 \
  --d_model 512 \
  --nhead 8 \
  --num_encoder_layers 2 \
  --num_decoder_layers 2 \
  --dim_feedforward 2048 \
  --dropout 0.5 \
  --embedding_dropout 0.1 \
  --source_embedding_activation softsign \
  --target_embedding_dim 300 \
  --target_embedding_activation softsign \
  --position_max_len 5000 \
  --gaslt_num_keys 7 \
  --gaslt_similarity_weight 1.0 \
  --semantic_model sentence-transformers/distiluse-base-multilingual-cased-v1 \
  --semantic_cache "${project_dir}/assets/gaslt/communication_sentence_similarity.pt" \
  --batch_size "${auslan_batch_size}" \
  --length_bucket_batches true \
  --grad_accum_steps "${auslan_grad_accum_steps}" \
  --lr 0.0005 \
  --weight_decay 0.001 \
  --epochs "${auslan_epochs}" \
  --lr_schedule plateau \
  --plateau_patience 9 \
  --plateau_factor 0.5 \
  --lr_min 0.0000001 \
  --label_smoothing 0.4 \
  --loss_normalization batch \
  --grad_clip 1.0 \
  --num_workers 0 \
  --device auto \
  --reference_bleu4 9.95 \
  --gate_fraction 0.9 \
  --out_dir "${run_dir}/out" \
  ${resume_args[@]+"${resume_args[@]}"} 2>&1 | tee -a "${run_dir}/train.log"
