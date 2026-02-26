
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

try:
    from model.normalization import RMSNorm
    from model.feedforward import SwiGLUFFN
    from model.attention import MultiHeadLatentAttention
except ModuleNotFoundError:
    from normalization import RMSNorm
    from feedforward import SwiGLUFFN
    from attention import MultiHeadLatentAttention


# ════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class NanoAgentConfig:
    """
    Configuration for NanoAgent-1B.

    Each value is chosen deliberately — see comments for reasoning.
    """
    # ── Model dimensions ──────────────────────────────────────────────
    hidden_dim: int = 2048       
    n_layers: int = 22         
    n_heads: int = 32          
    vocab_size: int = 32000    

    # ── SwiGLU FFN ────────────
    ffn_dim: int = 5632         
                                

    # ── MLA (Multi-Head Latent Attention) ─────────────────────────────
    d_c: int = 512              
                                 
                                 
                                
    d_c_q: int = 1536           
                                 
    d_rope: int = 64             
                                 

    # ── Sequence length ───────────────────────────────────────────────
    max_seq_len: int = 2048      
                                 

    # ── RoPE ──────────────────────────────────────────────────────────
    rope_theta: float = 500000.0 
                                 

    # ── MTP (Multi-Token Prediction) ──────────────────────────────────
    mtp_depth: int = 1           
                                 
    mtp_enabled: bool = True     

    # ── Training ──────────────────────────────────────────────────────
    dropout: float = 0.0         
                                 
                                 
                                 
                                 
                                 

    # ── Normalization ─────────────────────────────────────────────────
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
            d * d_cq                      
            + d_cq * (h * d_h)            
            + d_cq * (h * d_r)            
            + d * d_c                     
            + d_c * (h * d_h)             
            + d_c * (h * d_h)             
            + d * d_r                     
            + (h * d_h) * d               
            + d_cq                        
            + d_c                         
        )

        ffn_per_layer = 3 * d * f

        norms_per_layer = 2 * d  

        embedding = V * d

        final_norm = d

        mtp = d * d + V * d + d  

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
    """
    One Transformer block: RMSNorm → MLA → residual → RMSNorm → FFN → residual
    """

    def __init__(self, config: NanoAgentConfig):
        super().__init__()

        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        self.attention = MultiHeadLatentAttention(
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            d_c=config.d_c,
            d_c_q=config.d_c_q,
            d_rope=config.d_rope,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )

        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        # SwiGLU Feed-Forward Network
        self.ffn = SwiGLUFFN(config.hidden_dim, config.ffn_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: Causal attention mask

        Returns:
            Output tensor of same shape.
        """
        x = x + self.attention(self.attn_norm(x), mask=mask)

        x = x + self.ffn(self.ffn_norm(x))

        return x


class MTPModule(nn.Module):

    def __init__(self, config: NanoAgentConfig):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.proj = nn.Linear(2 * config.hidden_dim, config.hidden_dim, bias=False)

        self.norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        self.block = TransformerBlock(config)

        self.final_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        next_token_embeds: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        combined = torch.cat([hidden_states, next_token_embeds], dim=-1)

        h = self.proj(combined)
        h = self.norm(h)

        h = self.block(h, mask=mask)

        # Final normalization
        h = self.final_norm(h)

        return h


class NanoAgent(nn.Module):
    """
    NanoAgent-1B: A 1.1B parameter language model.

    Architecture: DeepSeek V3-inspired (MLA + MTP + SwiGLU + RMSNorm)
    Trained from scratch with GRPO for reasoning and tool use.
    """

    def __init__(self, config: NanoAgentConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

        if config.mtp_enabled:
            self.mtp = MTPModule(config)
        else:
            self.mtp = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
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
        """
        Full forward pass of NanoAgent-1B.

        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        h = self.token_embedding(input_ids)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        )

        for layer in self.layers:
            h = layer(h, mask=causal_mask)

        h = self.final_norm(h)

        logits = self.lm_head(h)

        result = {"logits": logits}

        if targets is not None:
            # Cross-entropy loss for next-token prediction
            main_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),  
                targets.view(-1),                          
                ignore_index=-1, 
            )
            result["loss"] = main_loss
            result["total_loss"] = main_loss

            if self.mtp is not None and seq_len > 2:
                next_token_ids = input_ids[:, 1:]  # [B, C, D, E]
                next_token_embeds = self.token_embedding(next_token_ids)

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
        """
        Generate text autoregressively.
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]

            # Forward pass 
            result = self(idx_cond)
            logits = result["logits"]

            logits = logits[:, -1, :] / temperature 

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# # ════════════════════════════════════════════════════════════════════════════
# # TESTING
# # ════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     print("=" * 70)
#     print("  NanoAgent-1B — Full Model Verification")
#     print("=" * 70)

#     config = NanoAgentConfig()

#     # Print parameter estimates
#     print("\n📊 Parameter Estimates:")
#     estimates = config.estimate_params()
#     for key, val in estimates.items():
#         print(f"   {key:25s}: {val:>15,}")
#     print(f"   {'':25s}  ({estimates['total']/1e9:.2f}B parameters)")

#     # Create model
#     print("\n🔨 Creating model...")
#     model = NanoAgent(config)

#     # Actual parameter count
#     actual_params = sum(p.numel() for p in model.parameters())
#     # Account for weight tying (embedding counted once, not twice)
#     unique_params = actual_params
#     print(f"   Actual parameters: {actual_params:,}")
#     print(f"   Estimated:         {estimates['total']:,}")

#     # Test 1: Forward pass
#     print("\n📋 Test 1: Forward pass (training mode)")
#     batch_size = 2
#     seq_len = 64
#     input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
#     targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

#     result = model(input_ids, targets=targets)
#     print(f"   Logits shape: {result['logits'].shape}")
#     print(f"   Main loss:    {result['loss'].item():.4f}")
#     if "mtp_loss" in result:
#         print(f"   MTP loss:     {result['mtp_loss'].item():.4f}")
#     print(f"   Total loss:   {result['total_loss'].item():.4f}")
#     print(f"   ✅ Forward pass works")

#     # Expected initial loss ≈ ln(vocab_size) = ln(32000) ≈ 10.37
#     expected_loss = math.log(config.vocab_size)
#     print(f"   Expected initial loss: ~{expected_loss:.2f} (random predictions)")
#     assert abs(result["loss"].item() - expected_loss) < 1.0, \
#         f"Initial loss {result['loss'].item():.2f} too far from expected {expected_loss:.2f}"
#     print(f"   ✅ Initial loss is in expected range")

#     # Test 2: Backward pass
#     print("\n📋 Test 2: Backward pass (gradient computation)")
#     result["total_loss"].backward()
#     grad_params = sum(1 for p in model.parameters() if p.grad is not None)
#     total_p = sum(1 for p in model.parameters())
#     print(f"   {grad_params}/{total_p} parameters received gradients")
#     print(f"   ✅ Backward pass works")

#     # Test 3: Generation
#     print("\n📋 Test 3: Generation (inference mode)")
#     model.eval()
#     prompt = torch.randint(0, config.vocab_size, (1, 5))  # 5-token prompt
#     generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
#     print(f"   Prompt length:    {prompt.shape[1]}")
#     print(f"   Generated length: {generated.shape[1]}")
#     assert generated.shape[1] == 15  # 5 + 10
#     print(f"   ✅ Generation works")

#     # Test 4: Memory footprint
#     print("\n📋 Test 4: Memory footprint")
#     param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
#     buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
#     print(f"   Parameters: {param_bytes / 1e9:.2f} GB (FP32)")
#     print(f"   Buffers:    {buffer_bytes / 1e6:.2f} MB (RoPE tables)")
#     print(f"   BF16 size:  {param_bytes / 2 / 1e9:.2f} GB (training)")
#     print(f"   ✅ Fits in GPU memory (4× RTX 4090 = 96GB)")

#     # Architecture summary
#     print(f"\n{'='*70}")
#     print(f"  ✅ NanoAgent-1B — All Tests Passed!")
#     print(f"{'='*70}")
#     print(f"  Architecture: DeepSeek V3-inspired dense Transformer")
#     print(f"  Parameters:   {actual_params/1e9:.2f}B")
#     print(f"  Components:")
#     print(f"    • MLA (Multi-Head Latent Attention) — DeepSeek V2/V3")
#     print(f"    • MTP (Multi-Token Prediction)      — DeepSeek V3")
#     print(f"    • SwiGLU FFN                        — Llama 3 / PaLM")
#     print(f"    • RMSNorm                           — Universal")
#     print(f"    • RoPE (θ=500K)                     — Universal")
#     print(f"    • Weight tying                      — Universal")
#     print(f"  Training: SFT → GRPO                 — DeepSeek R1")
#     print(f"{'='*70}")