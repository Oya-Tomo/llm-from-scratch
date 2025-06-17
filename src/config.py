from dataclasses import dataclass

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}


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

    @staticmethod
    def get_124m_config():
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


if __name__ == "__main__":
    gpt_config = GPTConfig.get_124m_config()
    print(gpt_config)
