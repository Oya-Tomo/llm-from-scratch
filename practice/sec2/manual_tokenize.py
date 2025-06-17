import urllib.request

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)
file_path = "data/the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

import re

text = "Hello, world. Is this-- a test?"


def split(text):
    spans = re.split(r"([,.:;?_!\"()']|--|\s+)", text)
    spans = [item.strip() for item in spans if item.strip()]
    return spans


print("Total number:", len(split(raw_text)))
print(split(raw_text)[:10])

all_words = sorted(set(split(raw_text)))
vocab_size = len(all_words)
print("Vocabulary size:", vocab_size)


def generate_vocab(text) -> tuple[dict, dict]:
    all_words = sorted(set(split(text)))
    enc_vocab = {token: num for num, token in enumerate(all_words)}
    dec_vocab = {num: token for num, token in enumerate(all_words)}
    return enc_vocab, dec_vocab


enc_vocab, dec_vocab = generate_vocab(raw_text)
print("Encoding vocabulary:", enc_vocab)
print("Decoding vocabulary:", dec_vocab)
