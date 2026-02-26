
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from model.normalization import RMSNorm
except ModuleNotFoundError:
    from normalization import RMSNorm


# ════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION: RoPE (Rotary Position Embeddings)
# ════════════════════════════════════════════════════════════════════════════

def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta_base: float = 500000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cosine and sine tables for RoPE.

    """

    freqs = 1.0 / (
        theta_base ** (torch.arange(0, dim, 2, device=device).float() / dim)
    )

    t = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(t, freqs)

    cos_cached = torch.cos(angles)
    sin_cached = torch.sin(angles)

    return cos_cached, sin_cached


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to input tensor.

    """
    seq_len = x.shape[-2]

    cos = cos[:seq_len]  
    sin = sin[:seq_len]  
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x_real = x_pairs[..., 0]  
    x_imag = x_pairs[..., 1]  

    while cos.dim() < x_real.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    # Apply rotation
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    out = torch.stack([out_real, out_imag], dim=-1)
    out = out.reshape(*x.shape)

    return out.type_as(x)


# ════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION: Multi-Head Latent Attention (MLA)
# ════════════════════════════════════════════════════════════════════════════

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek V2/V3.

    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        n_heads: int = 32,
        d_c: int = 512,        # KV compression dim (latent dim)
        d_c_q: int = 1536,     # Query compression dim
        d_rope: int = 64,      # RoPE dimension (per head)
        max_seq_len: int = 4096,
        rope_theta: float = 500000.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads  # 2048 // 32 = 64 (content head dim)
        self.d_c = d_c
        self.d_c_q = d_c_q
        self.d_rope = d_rope

      
        self.d_head_full = self.d_head + d_rope

      
        self.w_dq = nn.Linear(hidden_dim, d_c_q, bias=False)

        self.q_norm = RMSNorm(d_c_q)

        self.w_uq = nn.Linear(d_c_q, n_heads * self.d_head, bias=False)

        self.w_qr = nn.Linear(d_c_q, n_heads * d_rope, bias=False)
        self.w_dkv = nn.Linear(hidden_dim, d_c, bias=False)

        self.kv_norm = RMSNorm(d_c)

        self.w_uk = nn.Linear(d_c, n_heads * self.d_head, bias=False)

        self.w_uv = nn.Linear(d_c, n_heads * self.d_head, bias=False)

        self.w_kr = nn.Linear(hidden_dim, d_rope, bias=False)
        self.w_o = nn.Linear(n_heads * self.d_head, hidden_dim, bias=False)
        cos, sin = precompute_rope_frequencies(
            dim=d_rope,
            max_seq_len=max_seq_len,
            theta_base=rope_theta,
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        MLA Forward Pass.
        """
        batch_size, seq_len, _ = x.shape
        c_q = self.w_dq(x)
        c_q = self.q_norm(c_q)

        q_content = self.w_uq(c_q)

        q_rope = self.w_qr(c_q)

        q_content = q_content.view(batch_size, seq_len, self.n_heads, self.d_head)
        q_content = q_content.transpose(1, 2)

        q_rope = q_rope.view(batch_size, seq_len, self.n_heads, self.d_rope)
        q_rope = q_rope.transpose(1, 2)
        q_rope = apply_rope(q_rope, self.rope_cos, self.rope_sin)
        q = torch.cat([q_content, q_rope], dim=-1)

        c_kv = self.w_dkv(x)

        # Normalize
        c_kv_normed = self.kv_norm(c_kv)

        k_content = self.w_uk(c_kv_normed)
        k_content = k_content.view(batch_size, seq_len, self.n_heads, self.d_head)
        k_content = k_content.transpose(1, 2)

        v = self.w_uv(c_kv_normed)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.transpose(1, 2)

        k_rope = self.w_kr(x)

        # Reshape for heads: the rope key is SHARED across all heads
        k_rope = k_rope.unsqueeze(2).transpose(1, 2)

        k_rope = apply_rope(k_rope, self.rope_cos, self.rope_sin)

        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)

        # Concatenate content + rope for full key
        k = torch.cat([k_content, k_rope], dim=-1)

        scale = 1.0 / math.sqrt(self.d_head_full)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(q)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_head)

        # Final output projection
        output = self.w_o(attn_output)

        return output




# # ════════════════════════════════════════════════════════════════════════════
# # TESTING
# # ════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     print("=" * 60)
#     print("  Multi-Head Latent Attention — Implementation Verification")
#     print("=" * 60)

#     # Model config (our 1.1B model)
#     hidden_dim = 2048
#     n_heads = 32
#     d_c = 512
#     d_c_q = 1536
#     d_rope = 64
#     seq_len = 128  # Short for testing

#     mla = MultiHeadLatentAttention(
#         hidden_dim=hidden_dim,
#         n_heads=n_heads,
#         d_c=d_c,
#         d_c_q=d_c_q,
#         d_rope=d_rope,
#     )

#     # Test 1: Basic forward pass
#     print("\n📋 Test 1: Basic forward pass")
#     x = torch.randn(2, seq_len, hidden_dim)

#     # Create causal mask
#     causal_mask = torch.triu(
#         torch.full((seq_len, seq_len), float('-inf')),
#         diagonal=1,
#     )

#     y = mla(x, mask=causal_mask)
#     print(f"   Input shape:  {x.shape}")
#     print(f"   Output shape: {y.shape}")
#     assert x.shape == y.shape, f"Shape mismatch: {x.shape} != {y.shape}"
#     print(f"   ✅ Input and output shapes match")

#     # Test 2: Parameter count
#     print("\n📋 Test 2: Parameter count")
#     total_params = sum(p.numel() for p in mla.parameters())
#     print(f"   Total parameters: {total_params:,}")

#     # Breakdown
#     param_breakdown = {}
#     for name, p in mla.named_parameters():
#         param_breakdown[name] = p.numel()
#         print(f"     {name:30s}: {p.numel():>10,}")
#     print(f"   {'TOTAL':30s}: {total_params:>10,}")

#     # Test 3: KV Cache size comparison
#     print("\n📋 Test 3: KV Cache size comparison (per token, per layer)")
#     mha_cache = 2 * n_heads * (hidden_dim // n_heads)  # Full MHA
#     gqa_cache = 2 * 8 * (hidden_dim // n_heads)  # GQA with 8 KV heads
#     mqa_cache = 2 * (hidden_dim // n_heads)  # MQA
#     mla_cache = d_c + d_rope  # MLA

#     print(f"   MHA:  {mha_cache:,} values")
#     print(f"   GQA:  {gqa_cache:,} values (4× smaller than MHA)")
#     print(f"   MQA:  {mqa_cache:,} values ({mha_cache//mqa_cache}× smaller than MHA)")
#     print(f"   MLA:  {mla_cache:,} values ({mha_cache/mla_cache:.0f}× smaller than MHA) ✅")
#     print(f"   MLA achieves {mha_cache/mla_cache:.0f}× compression with BETTER quality!")

#     # Test 4: Causal masking
#     print("\n📋 Test 4: Causal masking verification")
#     x_small = torch.randn(1, 4, hidden_dim)
#     mask_small = torch.triu(
#         torch.full((4, 4), float('-inf')),
#         diagonal=1,
#     )
#     y1 = mla(x_small, mask=mask_small)
#     # Change token 4 — it shouldn't affect token 1's output
#     x_modified = x_small.clone()
#     x_modified[0, 3, :] = torch.randn(hidden_dim)
#     y2 = mla(x_modified, mask=mask_small)
#     # Tokens 1-3 should be identical (causal = can't see token 4)
#     diff = (y1[0, :3, :] - y2[0, :3, :]).abs().max().item()
#     print(f"   Max diff in tokens 0-2 after changing token 3: {diff:.2e}")
#     assert diff < 1e-5, f"Causal masking broken! diff={diff}"
#     print(f"   ✅ Causal masking works (future tokens don't leak)")

#     # Test 5: Gradient flow
#     print("\n📋 Test 5: Gradient flow")
#     x_grad = torch.randn(1, seq_len, hidden_dim, requires_grad=True)
#     y_grad = mla(x_grad, mask=causal_mask)
#     loss = y_grad.sum()
#     loss.backward()
#     assert x_grad.grad is not None
#     grad_params = sum(1 for p in mla.parameters() if p.grad is not None)
#     total_p = sum(1 for p in mla.parameters())
#     print(f"   {grad_params}/{total_p} parameters received gradients")
#     assert grad_params == total_p
#     print(f"   ✅ All parameters receive gradients")

#     # Test 6: RoPE verification
#     print("\n📋 Test 6: RoPE position encoding")
#     # RoPE should make attention position-dependent.
#     # We use DIFFERENT tokens per position so V vectors differ —
#     # otherwise identical V vectors produce identical outputs regardless
#     # of attention weights (since softmax sums to 1).
#     #
#     # Test: swap positions of two tokens → output should change
#     x_a = torch.randn(1, 4, hidden_dim)
#     x_b = x_a.clone()
#     x_b[0, [1, 2]] = x_b[0, [2, 1]]  # Swap positions 1 and 2
#     y_a = mla(x_a, mask=mask_small)
#     y_b = mla(x_b, mask=mask_small)
#     # Position 3 attends to all — swapping earlier tokens should change its output
#     diff = (y_a[0, 3, :] - y_b[0, 3, :]).abs().mean().item()
#     print(f"   Output diff at pos 3 after swapping pos 1,2: {diff:.6f}")
#     assert diff > 1e-6, "RoPE not encoding position!"
#     print(f"   ✅ RoPE makes position-swapped inputs produce different outputs")

#     print(f"\n{'='*60}")
#     print(f"  ✅ All tests passed! MLA is ready.")
#     print(f"  Next: architecture.py (full model + MTP)")
#     print(f"{'='*60}")