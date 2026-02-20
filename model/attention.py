import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from model.normalization import RMSNorm

def precompute_rope_frequencies(dim: int,
                                max_seq_len: int,
                                theta_base:float = 500000.0,
                                device:torch.device =None,) -> tuple[torch.Tensor, torch.Tensor]:
    
    freqs = 1.0 / (
        theta_base ** (torch.arange(0, dim,2,device=device).float() / dim)
    )

    t = torch.arange(max_seq_len, device=device).float()

    angles = torch.outer(t,freqs)

    cos_cached = torch.cos(angles)
    sin_cached = torch.sin(angles)

    return cos_cached, sin_cached

def apply_rope(
        x:torch.Tensor,
        cos:torch.Tensor,
        sin:torch.Tensor,
)-> torch.Tensor:
    
    seq_len = x.shape[-2]

    cos = cos[:seq_len]
    sin = sin[:seq_len]

    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)

    x_real = x_pairs[...,0]
    x_imag = x_pairs[..., 1]

    while cos.dim() < x_real.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # Apply rotation
    # This is the 2D rotation: [x_real, x_imag] × rotation_matrix
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    out = torch.stack([out_real, out_imag], dim=-1)
    out = out.reshape(*x.shape)

    return out.type_as(x)

class MultiHeadLatentAttention(nn.Module):
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
        # Precompute cos/sin tables for RoPE
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

        batch_size, seq_len, _ = x.shape
        c_q = self.w_dq(x)

        c_q = self.q_norm(c_q)

        q_content = self.w_uq(c_q)

        q_rope = self.w_qr(c_q)

        # Reshape to separate heads
        q_content = q_content.view(batch_size, seq_len, self.n_heads, self.d_head)
        q_content = q_content.transpose(1, 2)

        q_rope = q_rope.view(batch_size, seq_len, self.n_heads, self.d_rope)
        q_rope = q_rope.transpose(1, 2)
        # Both: (B, n_heads, S, d_*)

        # Apply RoPE to query rope component
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

        k_rope = k_rope.unsqueeze(2).transpose(1, 2)

        # Apply RoPE to key rope component
        k_rope = apply_rope(k_rope, self.rope_cos, self.rope_sin)

        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)

        k = torch.cat([k_content, k_rope], dim=-1)

        scale = 1.0 / math.sqrt(self.d_head_full)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask (prevent attending to future tokens)
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(q)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_head)

        output = self.w_o(attn_output)

        return output