import torch
import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(tokens: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    tokens = tokens.squeeze(0).tolist()
    return tokenizer.decode(tokens)
