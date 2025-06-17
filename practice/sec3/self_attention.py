import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super(SelfAttention_v1, self).__init__()
        self.W_q = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
        self.W_k = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
        self.W_v = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        querys = x @ self.W_q
        keys = x @ self.W_k
        values = x @ self.W_v

        attn_scores = querys @ keys.T
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super(SelfAttention_v2, self).__init__()
        self.W_q = nn.Linear(d_in, d_out)
        self.W_k = nn.Linear(d_in, d_out)
        self.W_v = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        querys = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attn_scores = querys @ keys.T
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)

        context_vec = attn_weights @ values
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
    )
    torch.manual_seed(123)
    d_in = inputs.shape[-1]
    d_out = 2
    self_attention = SelfAttention_v1(d_in, d_out)
    context_vec = self_attention(inputs)
    print("Context vector (v1):", context_vec)

    self_attention_v2 = SelfAttention_v2(d_in, d_out)
    context_vec_v2 = self_attention_v2(inputs)
    print("Context vector (v2):", context_vec_v2)
