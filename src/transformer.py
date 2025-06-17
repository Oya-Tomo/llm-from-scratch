import torch
from torch import nn

from config import GPTConfig
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm


class Transformer(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_len=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.rand((2, 4, 768))
    block = Transformer(GPTConfig.get_124m_config())
    output = block(x)

    print(x.shape)  # Expected input shape: (2, 4, 768)
    print(output.shape)  # Expected output shape: (2, 4, 768)
