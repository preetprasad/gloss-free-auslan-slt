"""LLM decoder over VQ sign tokens (Stage B).

Architecture:
  VQ codes (B, T) -> VQ code embeddings -> LLM encoder-decoder (mT5) -> English text

The VQ tokens are treated as a new "language" that the LLM translates to English.
We extend the mT5 tokenizer with VQ tokens and use LoRA on the decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5Config, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from peft import LoraConfig, get_peft_model, TaskType

from config import Config
from slt.sign_vq import VQTokeniser, build_vq_tokeniser


class VQCodeEmbedder(nn.Module):
    """Embed VQ codes into the LLM's embedding space."""

    def __init__(self, num_codes: int, code_dim: int, llm_dim: int, num_codebooks: int = 1):
        super().__init__()
        self.num_codebooks = num_codebooks
        if num_codebooks > 1:
            # Residual VQ: separate embedding per codebook, summed
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_codes, code_dim) for _ in range(num_codebooks)
            ])
            self.proj = nn.Linear(code_dim, llm_dim)
        else:
            self.embedding = nn.Embedding(num_codes, llm_dim)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (B, T) for single codebook, (B, T, K) for residual VQ
        Returns: (B, T, llm_dim)
        """
        if self.num_codebooks > 1:
            # codes: (B, T, K)
            embedded = sum(
                self.embeddings[k](codes[:, :, k]) for k in range(self.num_codebooks)
            )
            return self.proj(embedded)
        else:
            return self.embedding(codes)


class VQPoseEncoder(nn.Module):
    """Frozen VQ tokeniser encoder + VQ layer for Stage B inference."""

    def __init__(self, vq_tokeniser: VQTokeniser):
        super().__init__()
        self.encoder = vq_tokeniser.encoder
        self.vq = vq_tokeniser.vq
        self.eval()  # always frozen during Stage B

    @torch.no_grad()
    def forward(self, feats: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        """Returns discrete codes: (B, T) or (B, T, K)."""
        latent = self.encoder(feats, src_pad)
        _, codes, _ = self.vq(latent)
        return codes


class VQSignLLM(nn.Module):
    """Full model: pose/RGB features -> VQ encoder -> discrete codes -> LLM decoder -> English."""

    def __init__(self, feat_dim: int, cfg: Config, tokenizer, vq_tokeniser: Optional[VQTokeniser] = None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        # Stage A VQ tokeniser (frozen encoder + VQ)
        if vq_tokeniser is not None:
            self.vq_encoder = VQPoseEncoder(vq_tokeniser)
        else:
            # Will be loaded from checkpoint
            self.vq_encoder = None

        # Code embedder: maps VQ codes -> LLM embedding space
        mt5_config = MT5Config.from_pretrained(cfg.vq_llm_name)
        llm_dim = mt5_config.d_model
        self.code_embedder = VQCodeEmbedder(
            num_codes=cfg.vq_codebook_size,
            code_dim=cfg.vq_code_dim,
            llm_dim=llm_dim,
            num_codebooks=cfg.vq_num_codebooks,
        )

        # LLM with LoRA
        base_model = MT5ForConditionalGeneration.from_pretrained(cfg.vq_llm_name)
        for param in base_model.parameters():
            param.requires_grad_(False)

        target_modules = [m.strip() for m in cfg.vq_lora_target_modules.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.vq_lora_r,
            lora_alpha=cfg.vq_lora_alpha,
            lora_dropout=cfg.vq_lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.llm = get_peft_model(base_model, lora_config)
        self.llm.print_trainable_parameters()

        self.llm_dim = llm_dim

        # Store VQ encoder checkpoint path for loading
        self.vq_ckpt_path = cfg.vq_encoder_ckpt

    def load_vq_encoder(self, ckpt_path: str, device: torch.device):
        """Load VQ tokeniser from Stage A checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        vq_cfg = Config(**ckpt.get("cfg", {}))
        vq_tokeniser = build_vq_tokeniser(vq_cfg)
        vq_tokeniser.load_state_dict(ckpt["model"])
        vq_tokeniser.to(device)
        self.vq_encoder = VQPoseEncoder(vq_tokeniser)
        print(f"[VQ-LLM] Loaded VQ encoder from {ckpt_path}")

    def forward(
        self,
        feats: torch.Tensor,
        src_pad: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
    ):
        """
        feats: (B, T, F) raw pose/RGB features
        src_pad: (B, T) True=pad
        tgt_ids: (B, L) target token IDs
        tgt_attn_mask: (B, L) 1=valid
        """
        # Encode to discrete VQ codes
        if self.vq_encoder is None:
            raise RuntimeError("VQ encoder not loaded. Call load_vq_encoder() first.")
        codes = self.vq_encoder(feats, src_pad)  # (B, T) or (B, T, K)

        # Embed codes into LLM space
        code_embeds = self.code_embedder(codes)  # (B, T_codes, llm_dim)

        # Create encoder outputs for mT5
        encoder_outputs = BaseModelOutput(last_hidden_state=code_embeds)
        encoder_attention_mask = (~src_pad[:, :codes.shape[1]]).long() if codes.shape[1] != src_pad.shape[1] else (~src_pad).long()

        # Adjust attention mask if codes are downsampled
        if codes.shape[1] != src_pad.shape[1]:
            # Downsample mask to match codes length
            ratio = src_pad.shape[1] // codes.shape[1]
            encoder_attention_mask = (~src_pad[:, ::ratio]).long()

        labels = tgt_ids.clone()
        labels[~tgt_attn_mask.bool()] = -100

        outputs = self.llm(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=tgt_attn_mask,
            return_dict=True,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        feats: torch.Tensor,
        src_pad: torch.Tensor,
        max_length: int = 64,
        num_beams: int = 5,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        self.eval()
        if self.vq_encoder is None:
            raise RuntimeError("VQ encoder not loaded. Call load_vq_encoder() first.")

        codes = self.vq_encoder(feats, src_pad)
        code_embeds = self.code_embedder(codes)

        encoder_outputs = BaseModelOutput(last_hidden_state=code_embeds)
        if codes.shape[1] != src_pad.shape[1]:
            ratio = src_pad.shape[1] // codes.shape[1]
            encoder_attention_mask = (~src_pad[:, ::ratio]).long()
        else:
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


def build_vq_sign_llm(feat_dim: int, cfg: Config, tokenizer, vq_tokeniser: VQTokeniser = None) -> VQSignLLM:
    return VQSignLLM(feat_dim, cfg, tokenizer, vq_tokeniser)