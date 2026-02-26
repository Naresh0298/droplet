import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()

        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# # ════════════════════════════════════════════════════════════════════════════
# # TESTING — Verify the implementation
# # ════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     print("=" * 60)
#     print("  SwiGLU FFN — Implementation Verification")
#     print("=" * 60)

#     hidden_dim = 2048
#     ffn_dim = 5632

#     ffn = SwiGLUFFN(hidden_dim, ffn_dim)

#     # Test 1: Basic forward pass
#     print("\n📋 Test 1: Basic forward pass")
#     x = torch.randn(2, 10, hidden_dim)  # (batch=2, seq_len=10, dim=2048)
#     y = ffn(x)
#     print(f"   Input shape:  {x.shape}")
#     print(f"   Output shape: {y.shape}")
#     assert x.shape == y.shape, "Shape mismatch!"
#     print(f"   ✅ Input and output shapes match")

#     # Test 2: Parameter count
#     print("\n📋 Test 2: Parameter count")
#     total_params = sum(p.numel() for p in ffn.parameters())
#     expected = 3 * hidden_dim * ffn_dim  # 3 matrices
#     print(f"   Total parameters: {total_params:,}")
#     print(f"   Expected:         {expected:,}")
#     assert total_params == expected, f"Expected {expected}, got {total_params}"
#     print(f"   ✅ Correct parameter count")

#     # Test 3: No bias parameters
#     print("\n📋 Test 3: No bias parameters")
#     has_bias = any("bias" in name for name, _ in ffn.named_parameters())
#     print(f"   Has bias: {has_bias}")
#     assert not has_bias, "Found unexpected bias parameters!"
#     print(f"   ✅ No bias (correct for modern LLMs)")

#     # Test 4: BF16 compatibility
#     print("\n📋 Test 4: BF16 mixed precision")
#     ffn_bf16 = SwiGLUFFN(hidden_dim, ffn_dim).to(torch.bfloat16)  # Fresh copy
#     x_bf16 = x.to(torch.bfloat16)
#     y_bf16 = ffn_bf16(x_bf16)
#     assert y_bf16.dtype == torch.bfloat16
#     assert not torch.isnan(y_bf16).any()
#     print(f"   ✅ BF16 works correctly, no NaN")

#     # Test 5: Gradient flow
#     print("\n📋 Test 5: Gradient flow")
#     x_grad = torch.randn(2, 10, hidden_dim, requires_grad=True)
#     y_grad = ffn(x_grad)
#     loss = y_grad.sum()
#     loss.backward()
#     assert x_grad.grad is not None
#     assert all(p.grad is not None for p in ffn.parameters())
#     print(f"   ✅ Gradients flow through all 3 matrices")

#     # Test 6: Gating behavior visualization
#     print("\n📋 Test 6: Gating behavior")
#     with torch.no_grad():
#         x_viz = torch.randn(1, 1, hidden_dim)
#         gate = F.silu(ffn.w_gate(x_viz))
#         up = ffn.w_up(x_viz)
#         gated = gate * up

#         # How many neurons are effectively "off" (gate ≈ 0)?
#         gate_values = gate.abs().squeeze()
#         suppressed = (gate_values < 0.01).sum().item()
#         active = (gate_values > 0.1).sum().item()
#         print(f"   FFN dim: {ffn_dim}")
#         print(f"   Active neurons (|gate| > 0.1):     {active:,} ({active/ffn_dim*100:.1f}%)")
#         print(f"   Suppressed neurons (|gate| < 0.01): {suppressed:,} ({suppressed/ffn_dim*100:.1f}%)")
#         print(f"   ✅ Gate is selectively activating neurons (this is the point!)")

#     # Parameter breakdown
#     print("\n📋 Parameter breakdown (per block):")
#     print(f"   W_gate: {hidden_dim} × {ffn_dim} = {hidden_dim * ffn_dim:>12,}")
#     print(f"   W_up:   {hidden_dim} × {ffn_dim} = {hidden_dim * ffn_dim:>12,}")
#     print(f"   W_down: {ffn_dim} × {hidden_dim} = {ffn_dim * hidden_dim:>12,}")
#     print(f"   Total per block:                   {3 * hidden_dim * ffn_dim:>12,}")
#     print(f"   Total for 22 blocks:               {22 * 3 * hidden_dim * ffn_dim:>12,}")

#     print(f"\n{'='*60}")
#     print(f"  ✅ All tests passed! SwiGLU FFN is ready.")
#     print(f"  Next: attention.py (MLA — the hardest part)")
#     print(f"{'='*60}")