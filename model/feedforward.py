import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):

    def __init__(self, hidden_dim: int, ffn_dim:int):
        super().__init__()

        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)

        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias = False)

        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x:torch.tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
    
    