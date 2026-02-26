import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight



# # ════════════════════════════════════════════════════════════════════════════
# # TESTING — Verify the implementation
# # ════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     print("=" * 60)
#     print("  RMSNorm — Implementation Verification")
#     print("=" * 60)

#     # Test 1: Basic functionality
#     print("\n📋 Test 1: Basic forward pass")
#     dim = 2048
#     norm = RMSNorm(dim)
#     x = torch.randn(2, 10, dim)  # (batch=2, seq_len=10, dim=2048)
#     y = norm(x)
#     print(f"   Input shape:  {x.shape}")
#     print(f"   Output shape: {y.shape}")
#     assert x.shape == y.shape, "Shape mismatch!"
#     print(f"   ✅ Shapes match")

#     # Test 2: Verify normalization behavior
#     print("\n📋 Test 2: Normalization behavior")
#     # After RMSNorm (with weight=1), the RMS of output should be ≈ 1.0
#     rms_before = torch.sqrt(x.pow(2).mean(-1)).mean().item()
#     rms_after = torch.sqrt(y.pow(2).mean(-1)).mean().item()
#     print(f"   RMS before norm: {rms_before:.4f}")
#     print(f"   RMS after norm:  {rms_after:.4f}")
#     assert abs(rms_after - 1.0) < 0.1, f"RMS should be ≈1.0, got {rms_after}"
#     print(f"   ✅ RMS ≈ 1.0 after normalization")

#     # Test 3: Works with BF16 (mixed precision)
#     print("\n📋 Test 3: BF16 mixed precision")
#     norm_bf16 = RMSNorm(dim).to(torch.bfloat16)  # Fresh copy
#     x_bf16 = x.to(torch.bfloat16)
#     y_bf16 = norm_bf16(x_bf16)
#     print(f"   Input dtype:  {x_bf16.dtype}")
#     print(f"   Output dtype: {y_bf16.dtype}")
#     assert y_bf16.dtype == torch.bfloat16, "Output should be BF16"
#     print(f"   ✅ BF16 works correctly")

#     # Test 4: No NaN with zero input (epsilon test)
#     print("\n📋 Test 4: Zero input (epsilon stability)")
#     x_zero = torch.zeros(1, 1, dim)
#     y_zero = norm(x_zero)
#     assert not torch.isnan(y_zero).any(), "NaN detected with zero input!"
#     print(f"   ✅ No NaN with zero input (epsilon works)")

#     # Test 5: Parameter count
#     print("\n📋 Test 5: Parameter count")
#     param_count = sum(p.numel() for p in norm.parameters())
#     print(f"   Parameters: {param_count:,} (expected: {dim:,})")
#     assert param_count == dim, f"Expected {dim} params, got {param_count}"
#     print(f"   ✅ Correct parameter count")

#     # Test 6: Gradient flow
#     print("\n📋 Test 6: Gradient flow")
#     x_grad = torch.randn(2, 10, dim, requires_grad=True)
#     y_grad = norm(x_grad)
#     loss = y_grad.sum()
#     loss.backward()
#     assert x_grad.grad is not None, "No gradient for input!"
#     assert norm.weight.grad is not None, "No gradient for weight!"
#     print(f"   ✅ Gradients flow through correctly")

#     # Compare with PyTorch LayerNorm (for understanding, not for use)
#     print("\n📋 Comparison: RMSNorm vs LayerNorm")
#     layer_norm = nn.LayerNorm(dim)
#     x_test = torch.randn(1, 1, dim)

#     # Time comparison (rough)
#     import time
#     n_iters = 1000

#     start = time.time()
#     for _ in range(n_iters):
#         _ = norm(x_test)
#     rms_time = time.time() - start

#     start = time.time()
#     for _ in range(n_iters):
#         _ = layer_norm(x_test)
#     ln_time = time.time() - start

#     speedup = (ln_time - rms_time) / ln_time * 100
#     print(f"   RMSNorm:   {rms_time*1000:.1f}ms ({n_iters} iterations)")
#     print(f"   LayerNorm: {ln_time*1000:.1f}ms ({n_iters} iterations)")
#     print(f"   Speedup:   {speedup:.1f}% faster")

#     print(f"\n{'='*60}")
#     print(f"  ✅ All tests passed! RMSNorm is ready.")
#     print(f"{'='*60}")