import torch
from torch import nn


from config import GPTConfig
from transformer import Transformer
from layer_norm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

        self.transformers = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.output_layer = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_batches, d_seq_len = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(d_seq_len, device=x.device))
        x = tok_emb + pos_emb
        x = self.dropout(x)

        x = self.transformers(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits


if __name__ == "__main__":
    cfg = GPTConfig.get_124m_config()
    model = GPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size - 1, (2, cfg.context_length))
    logits = model(x)
    print(logits.shape)
