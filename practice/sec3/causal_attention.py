import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_len: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_len = context_len

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(context_len, context_len),
                diagonal=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d_in = x.shape
        keys = self.W_k(x)  # (b, t, d_out)
        querys = self.W_q(x)  # (b, t, d_out)
        values = self.W_v(x)  # (b, t, d_out)

        attn_scores: torch.Tensor = querys @ keys.transpose(1, 2)  # (b, t, t)
        attn_scores.masked_fill_(self.mask.bool()[:t, :t] == 0, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.d_out**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, t, d_out)
        return context_vec


if __name__ == "__main__":
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    ).unsqueeze(0)

    torch.manual_seed(123)
    d_in = inputs.shape[-1]
    d_out = 2
    context_len = inputs.shape[1]
    dropout = 0.1

    causal_attention = CausalAttention(d_in, d_out, context_len, dropout)
    context_vec = causal_attention(inputs)

    print("Context vector:", context_vec)
    print("Context vector shape:", context_vec.shape)
