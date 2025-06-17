import torch
from torch import nn

from config import GPTConfig
from activation import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim * 4),
            GELU(),
            nn.Linear(cfg.emb_dim * 4, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__ == "__main__":
    ffn = FeedForward(GPTConfig.get_124m_config())

    x = torch.randn(3, 5, 768)  # (batch_size, seq_len, emb_dim)
    out = ffn(x)
    print(out.shape)
