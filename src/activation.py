import torch
from torch import nn


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


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("QT5Agg")

    relu = nn.ReLU()
    gelu = GELU()
    x = torch.linspace(-5, 5, 1000)
    plt.figure(figsize=(10, 5))
    for i, (y, label) in enumerate(zip([relu(x), gelu(x)], ["ReLU", "GELU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(label)
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid()
    plt.show()
