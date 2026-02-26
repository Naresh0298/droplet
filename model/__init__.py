"""
NanoAgent-1B Model Architecture.

DeepSeek V3-inspired dense Transformer with:
- Multi-Head Latent Attention (MLA) — DeepSeek V2/V3
- Multi-Token Prediction (MTP) — DeepSeek V3  
- SwiGLU Feed-Forward Network — Llama 3 / PaLM
- RMSNorm — Universal modern LLM component
- RoPE with θ=500K — Universal
"""

try:
    from model.normalization import RMSNorm
    from model.feedforward import SwiGLUFFN
    from model.attention import MultiHeadLatentAttention
    from model.architecture import NanoAgent, NanoAgentConfig, TransformerBlock, MTPModule
except ModuleNotFoundError:
    from normalization import RMSNorm
    from feedforward import SwiGLUFFN
    from attention import MultiHeadLatentAttention
    from architecture import NanoAgent, NanoAgentConfig, TransformerBlock, MTPModule

__all__ = [
    "RMSNorm",
    "SwiGLUFFN", 
    "MultiHeadLatentAttention",
    "NanoAgent",
    "NanoAgentConfig",
    "TransformerBlock",
    "MTPModule",
]