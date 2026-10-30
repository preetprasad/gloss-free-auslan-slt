"""Pose encoder -> adapter -> pretrained mBART decoder (BPE / subword targets).

The GFSLT-VLP-style route: use a PRETRAINED language decoder (mBART-large-50, which already knows
fluent English) with its own SUBWORD tokenizer, fed by our pose encoder through a learned adapter.
This directly fixes why the earlier LLM attempts failed:
  * mT5+LoRA decoder (7.66): word-level vocab + under-trained rank-16 LoRA over unaligned features.
  * LLM-VQ (0.0): degenerate discrete tokens.
Here: mBART BPE tokenizer (rich language prior), a trainable adapter + trainable pose encoder
(warm-started from our VLP/SWA checkpoint), LoRA across ALL mBART attention+FFN, and the correct
forced English BOS at generation. The mBART encoder is bypassed; its decoder cross-attends to the
adapted pose memory (encoder_outputs), exactly like a translation decoder.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model, TaskType

from slt.reference_model import build_model


class PoseMBART(nn.Module):
    def __init__(self, feat_dim: int, cfg, forced_bos_en: int):
        super().__init__()
        # Reuse our pose encoder (PoseSLT/ReferencePoseSLT.encode) so VLP/SWA checkpoints load.
        self.base = build_model(feat_dim, 8, cfg)
        self.forced_bos_en = forced_bos_en

        mbart = MBartForConditionalGeneration.from_pretrained(cfg.mbart_name)
        self.d_mbart = mbart.config.d_model
        self.adapter = nn.Sequential(
            nn.Linear(cfg.d_model, self.d_mbart), nn.LayerNorm(self.d_mbart), nn.Dropout(cfg.dropout))
        for p in mbart.parameters():           # freeze base mBART; train via LoRA
            p.requires_grad_(False)
        lc = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=cfg.mbart_lora_r,
                        lora_alpha=2 * cfg.mbart_lora_r, lora_dropout=0.05, bias="none",
                        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
        self.mbart = get_peft_model(mbart, lc)
        # Stage 2b ALIGN phase: freeze LoRA too, so only adapter + pose encoder learn to map into
        # the frozen pretrained decoder's space (GFSLT-VLP-style visual-language alignment).
        if not getattr(cfg, "mbart_train_lora", True):
            for n, p in self.mbart.named_parameters():
                if "lora" in n.lower():
                    p.requires_grad_(False)

    def _encode(self, src, src_pad):
        mem = self.base.encode(src, src_pad)          # (B, T, d_model)
        return self.adapter(mem)                       # (B, T, d_mbart)

    def forward(self, src, src_pad, labels):
        h = self._encode(src, src_pad)
        enc = BaseModelOutput(last_hidden_state=h)
        return self.mbart(encoder_outputs=enc,
                          attention_mask=(~src_pad).long(),
                          labels=labels)                # HF computes shifted-CE loss

    @torch.no_grad()
    def generate(self, src, src_pad, max_len=64, num_beams=5, length_penalty=1.0):
        h = self._encode(src, src_pad)
        enc = BaseModelOutput(last_hidden_state=h)
        return self.mbart.generate(encoder_outputs=enc,
                                   attention_mask=(~src_pad).long(),
                                   forced_bos_token_id=self.forced_bos_en,
                                   max_length=max_len, num_beams=num_beams,
                                   length_penalty=length_penalty, early_stopping=True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_pose_mbart(feat_dim, cfg, tokenizer):
    en = tokenizer.lang_code_to_id["en_XX"] if hasattr(tokenizer, "lang_code_to_id") \
        else tokenizer.convert_tokens_to_ids("en_XX")
    return PoseMBART(feat_dim, cfg, forced_bos_en=en)
