import torch

import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea?"

tokens = tokenizer.encode(text)
print("Tokenized text:", tokens)

embedding = torch.nn.Embedding(
    num_embeddings=tokenizer.n_vocab,
    embedding_dim=512,
)

token_embeddings = embedding(torch.tensor(tokens))
print("Token embeddings shape:", token_embeddings.shape)
