import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_len: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num-heads"

        self.num_heads = num_heads
        self.d_out = d_out
        self.d_out_head = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.output_layer = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_len, context_len),
                diagonal=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_batches, d_tokens, d_in = x.shape

        keys: torch.Tensor = self.W_k(x)
        queries: torch.Tensor = self.W_q(x)
        values: torch.Tensor = self.W_v(x)

        keys = keys.view(d_batches, d_tokens, self.num_heads, self.d_out_head)
        queries = queries.view(d_batches, d_tokens, self.num_heads, self.d_out_head)
        values = values.view(d_batches, d_tokens, self.num_heads, self.d_out_head)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:d_tokens, :d_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weight: torch.Tensor = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weight: torch.Tensor = self.dropout(attn_weight)

        context_vec = (attn_weight @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(d_batches, d_tokens, self.d_out)
        context_vec = self.output_layer(context_vec)
        return context_vec


if __name__ == "__main__":
    mha = MultiHeadAttention(4, 6, 10, 0.1, 3, True)

    x = torch.randn((2, 10, 4))
    y = mha(x)

    print(y.shape)
