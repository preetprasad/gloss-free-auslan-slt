"""Pose encoder + mT5 decoder with LoRA for gloss-free Auslan SLT.

Architecture:
  Pose (B,T,399) -> PoseEncoder (ReferencePoseSLT encoder) -> Bridge (256->768)
    -> mT5 decoder (LoRA) -> English tokens

The mT5 encoder is bypassed; pose features are fed via encoder_outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5Config
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from peft import LoraConfig, get_peft_model, TaskType

from .reference_model import ReferencePoseSLT
from .data import PAD
from config import Config


class PoseEncoder(nn.Module):
    """Wrapper around the ReferencePoseSLT encoder for clean composition."""

    def __init__(self, feat_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        vocab_size = 1000
        dummy_cfg = Config()
        dummy_cfg.model_arch = "signjoey_slt"
        dummy_cfg.d_model = d_model
        dummy_cfg.nhead = cfg.nhead
        dummy_cfg.num_encoder_layers = cfg.num_encoder_layers
        dummy_cfg.num_decoder_layers = cfg.num_decoder_layers
        dummy_cfg.dim_feedforward = cfg.dim_feedforward
        dummy_cfg.dropout = cfg.dropout
        dummy_cfg.embedding_dropout = cfg.embedding_dropout
        dummy_cfg.source_embedding_activation = cfg.source_embedding_activation
        dummy_cfg.target_embedding_dim = cfg.target_embedding_dim
        dummy_cfg.target_embedding_activation = cfg.target_embedding_activation
        dummy_cfg.position_max_len = cfg.position_max_len
        dummy_cfg.gaslt_num_keys = cfg.gaslt_num_keys
        dummy_cfg.visual_feature_dim = cfg.visual_feature_dim

        self.model = ReferencePoseSLT(feat_dim, vocab_size, dummy_cfg)

    def forward(self, src: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        return self.model.encode(src, src_pad)

    def load_state_dict(self, state_dict, strict: bool = True):
        encoder_prefixes = (
            "pose_adapter", "spatial_embedding", "source_position",
            "encoder_layers", "encoder_norm"
        )
        filtered = {k: v for k, v in state_dict.items()
                    if any(k.startswith(p) for p in encoder_prefixes)}
        return self.model.load_state_dict(filtered, strict=False)


class Bridge(nn.Module):
    """Project pose encoder d_model -> mT5 d_model (768 for mt5-base)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PoseLLM(nn.Module):
    """Full pose-to-text model: pose encoder + bridge + mT5 decoder (LoRA)."""

    def __init__(self, feat_dim: int, cfg: Config, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.pose_encoder = PoseEncoder(feat_dim, cfg)

        mt5_config = MT5Config.from_pretrained(cfg.llm_name)
        llm_d_model = mt5_config.d_model
        self.bridge = Bridge(cfg.d_model, llm_d_model)

        base_model = MT5ForConditionalGeneration.from_pretrained(cfg.llm_name)
        for param in base_model.parameters():
            param.requires_grad_(False)

        target_modules = [m.strip() for m in cfg.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.llm = get_peft_model(base_model, lora_config)
        self.llm.print_trainable_parameters()

        self.llm_d_model = llm_d_model

    def forward(
        self,
        src: torch.Tensor,
        src_pad: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
        return_dict: bool = True,
    ):
        memory = self.pose_encoder(src, src_pad)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=self.bridge(memory)
        )
        encoder_attention_mask = (~src_pad).long()

        labels = tgt_ids.clone()
        labels[~tgt_attn_mask.bool()] = -100

        outputs = self.llm(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=tgt_attn_mask,
            return_dict=return_dict,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_pad: torch.Tensor,
        max_length: int = 64,
        num_beams: int = 5,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        self.eval()
        memory = self.pose_encoder(src, src_pad)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=self.bridge(memory)
        )
        encoder_attention_mask = (~src_pad).long()

        generated = self.llm.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return generated

    def get_trainable_params(self):
        trainable = []
        frozen = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append((name, param))
            else:
                frozen.append((name, param))
        return trainable, frozen


def build_pose_llm(feat_dim: int, cfg: Config, tokenizer) -> PoseLLM:
    return PoseLLM(feat_dim, cfg, tokenizer)