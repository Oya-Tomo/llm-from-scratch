import torch


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

# Self-Attention Mechanism Example

query = inputs[0]

attn_scores = torch.zeros(inputs.shape[0])

for i in range(inputs.shape[0]):
    attn_scores[i] = torch.dot(query, inputs[i])

print("Attention scores:", attn_scores)
print("Attention weights:", torch.softmax(attn_scores, dim=0))


attn_scores = inputs @ inputs.T

print("Attention scores (matrix multiplication):", attn_scores)
print(
    "Attention weights (matrix multiplication):",
    torch.softmax(attn_scores, dim=1),
    attn_scores.shape,
)

# Scaled Dot-Product Attention

d_in = 3
d_out = 2

torch.manual_seed(123)

W_q = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_k = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_v = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

query = inputs @ W_q
key = inputs @ W_k
value = inputs @ W_v

print("Query:", query)
print("Key:", key)
print("Value:", value)
