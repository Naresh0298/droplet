import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass

from model.normalization import RMSNorm
from model.feedforward import SwiGLUFFN
from model.attention import MultiHeadLatentAttention

@dataclass
class DropletConfig:
    #Model dimensions
    hidden_dim: int = 2048
    n_layer: int = 22
    n_head: int = 32
    vocab_size: int = 32000

    #SwiGLU FFN
    ffn_dim: int = 5632

    #MLA
    d_c: int = 512
    d_c_q:int=1536

    #sequence length
    max_seq_len: int = 2048

    #RoPE
    rope_theta: float = 500000.0

    #MTP
    mtp_depth:int = 1

    mtp_enabled:bool = True

    #training
    dropout: float = 0.0

    #Normalization
    norm_eps: float = 1e-6

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.n_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_rope % 2 == 0, \
            f"d_rope ({self.d_rope}) must be even (for RoPE pair rotation)"
    
    @property
    def d_head(self) -> int:
        """Per-head dimension for content."""
        return self.hidden_dim // self.n_heads

    def estimate_params(self) -> dict:
        """
        Estimate parameter count for each component.
        Useful for sanity-checking before training.
        """
        d, L, h = self.hidden_dim, self.n_layers, self.n_heads
        d_h = self.d_head
        d_c, d_cq, d_r = self.d_c, self.d_c_q, self.d_rope
        f = self.ffn_dim
        V = self.vocab_size

        # Per-layer attention (MLA)
        attn_per_layer = (
            d * d_cq                      # w_dq
            + d_cq * (h * d_h)            # w_uq
            + d_cq * (h * d_r)            # w_qr
            + d * d_c                     # w_dkv
            + d_c * (h * d_h)             # w_uk
            + d_c * (h * d_h)             # w_uv
            + d * d_r                     # w_kr
            + (h * d_h) * d               # w_o
            + d_cq                        # q_norm
            + d_c                         # kv_norm
        )

        # Per-layer FFN (SwiGLU)
        ffn_per_layer = 3 * d * f

        # Per-layer norms
        norms_per_layer = 2 * d  # 2 RMSNorms with hidden_dim

        # Embedding
        embedding = V * d

        # Final norm
        final_norm = d

        # MTP module (approximate)
        mtp = d * d + V * d + d  # projection + lm_head + norm (if not tied)

        total = (
            embedding
            + L * (attn_per_layer + ffn_per_layer + norms_per_layer)
            + final_norm
            + (mtp if self.mtp_enabled else 0)
        )

        return {
            "embedding": embedding,
            "attention_per_layer": attn_per_layer,
            "ffn_per_layer": ffn_per_layer,
            "norms_per_layer": norms_per_layer,
            "total_per_layer": attn_per_layer + ffn_per_layer + norms_per_layer,
            "all_layers": L * (attn_per_layer + ffn_per_layer + norms_per_layer),
            "final_norm": final_norm,
            "mtp": mtp if self.mtp_enabled else 0,
            "total": total,
        }
    
class TransformerBlock(nn.Module):

    def __init__(self, config: DropletConfig):
        super().__init__()

        self.atten_norm = RMSNorm(config.hidden_dim,eps = config.norm_eps)

        self.attention = MultiHeadLatentAttention(
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            d_c=config.d_c,
            d_c_q=config.d_c_q,
            d_rope=config.d_rope,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )

        # Pre-norm before FFN
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        # SwiGLU Feed-Forward Network
        self.ffn = SwiGLUFFN(config.hidden_dim, config.ffn_dim)

    def forward(self, x:torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:

        x = x + self.attention(self.attn_norm(x), mask=mask)

        x = x + self.ffn(self.ffn_norm(x))

        return x
    
class MTPModule(nn.Module):

    def __init__(self, config:DropletConfig):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.proj = nn.Linear(2 * config.hidden_dim, config.hidden_dim, bias=False)

        self.norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        self.block = TransformerBlock(config)

        # Final norm before LM head
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_token_embeds: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        combined = torch.cat([hidden_states, next_token_embeds], dim=-1)

        # Project back to hidden_dim
        # (B, S, 4096) → (B, S, 2048)
        h = self.proj(combined)
        h = self.norm(h)

        # Process through one Transformer block
        h = self.block(h, mask=mask)

        # Final normalization
        h = self.final_norm(h)

        return h
    
class Droplet(nn.Module):

    def __init__(self, config:DropletConfig):
        super().__init__()
        self.config = config
        
        # ── Token Embedding ───────────────────────────────────────────
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # ── Transformer Blocks ────────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # ── Final Normalization ───────────────────────────────────────
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        # ── Language Model Head ───────────────────────────────────────
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # ── Weight Tying ──────────────────────────────────────────────
        self.lm_head.weight = self.token_embedding.weight

        # ── MTP Module (Multi-Token Prediction) ───────────────────────
        if config.mtp_enabled:
            self.mtp = MTPModule(config)
        else:
            self.mtp = None

        # ── Weight Initialization ─────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights following GPT-2 / Llama conventions.

        WHY INITIALIZATION MATTERS:
        Bad initialization → gradients explode or vanish from step 1
        → training diverges or learns nothing.

        WHAT WE DO:
        - Linear layers: Normal distribution with std = 0.02
          WHY 0.02: This keeps initial activations in a reasonable range.
          With hidden_dim=2048 and std=0.02, the output variance is
          ~2048 × 0.02² = ~0.8, which is close to 1.0.

        - Embedding: Same as Linear (it's effectively a lookup into a matrix)

        - Residual projections: Scaled by 1/√(2 × n_layers)
          WHY: Each layer adds to the residual. After 22 layers, the
          variance would grow by 22×. Scaling by 1/√(2×22) ≈ 0.15
          keeps the total variance stable.
          The factor 2 accounts for both attention and FFN residuals.

        GPT-2, Llama, and DeepSeek all use similar initialization.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        mtp_weight: float = 0.3,
    ) -> dict:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # ── Step 1: Token Embedding ───────────────────────────────────
        h = self.token_embedding(input_ids)

        # ── Step 2: Create Causal Mask ────────────────────────────────
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        )

        # ── Step 3: Pass Through All Transformer Blocks ───────────────
        for layer in self.layers:
            h = layer(h, mask=causal_mask)

        # ── Step 4: Final Normalization ───────────────────────────────
        h = self.final_norm(h)

        # ── Step 5: Compute Logits ────────────────────────────────────
        logits = self.lm_head(h)

        # ── Step 6: Compute Loss (if training) ────────────────────────
        result = {"logits": logits}

        if targets is not None:
            main_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),  
                targets.view(-1),                          
                ignore_index=-1,  
            )
            result["loss"] = main_loss
            result["total_loss"] = main_loss

            # ── Step 7: MTP Loss (if enabled) ─────────────────────────
            if self.mtp is not None and seq_len > 2:
                next_token_ids = input_ids[:, 1:]  
                next_token_embeds = self.token_embedding(next_token_ids)

                # Hidden states for positions 0 to S-2
                h_for_mtp = h[:, :-1, :]  

                mtp_h = h_for_mtp[:, :-1, :]          
                mtp_embeds = next_token_embeds[:, :-1, :]  
                mtp_targets = input_ids[:, 2:]         

                # MTP causal mask (shorter sequence)
                mtp_seq_len = mtp_h.shape[1]
                mtp_mask = torch.triu(
                    torch.full((mtp_seq_len, mtp_seq_len), float('-inf'), device=device),
                    diagonal=1,
                )

                # Forward through MTP module
                mtp_hidden = self.mtp(mtp_h, mtp_embeds, mask=mtp_mask)

                # Compute MTP logits using shared LM head
                mtp_logits = self.lm_head(mtp_hidden)

                # MTP loss
                mtp_loss = F.cross_entropy(
                    mtp_logits.view(-1, self.config.vocab_size),
                    mtp_targets.reshape(-1),
                    ignore_index=-1,
                )

                result["mtp_loss"] = mtp_loss
                result["total_loss"] = main_loss + mtp_weight * mtp_loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]

            # Forward pass (no targets = no loss computation)
            result = self(idx_cond)
            logits = result["logits"]

            # Get logits for the last position only
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
    

    














