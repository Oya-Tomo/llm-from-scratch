import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    config_title: str
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool

    def get_124m_config(self):
        return GPTConfig(
            config_title="GPT-124M",
            vocab_size=50257,
            context_length=1024,
            emb_dim=768,
            n_heads=12,
            n_layers=12,
            drop_rate=0.1,
            qkv_bias=True,
        )


class DummyGPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            [DummyTransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = DummyLayerNorm(cfg.emb_dim)
        self.output_layer = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_batches, d_seq_len = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(d_seq_len, device=x.device))
        x = token_emb + pos_emb
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder for transformer block logic
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder for transformer block logic
        return x


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


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


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
    ln = LayerNorm(emb_dim=5)
    out = ln(torch.randn(3, 5))
    print(out.mean(dim=-1))
    print(out.var(dim=-1, unbiased=False, keepdim=True))
