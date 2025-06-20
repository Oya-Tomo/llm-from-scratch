import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift


if __name__ == "__main__":
    ln = LayerNorm(emb_dim=5)
    out = ln(torch.randn(3, 5))
    print(out.mean(dim=-1))
    print(out.var(dim=-1, unbiased=False, keepdim=True))
