import tiktoken

file_path = "data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()


tokenizer = tiktoken.get_encoding("gpt2")

print("Total number of character:", len(raw_text))

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
)
nums = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Tokenized text:", nums)

text_recov = tokenizer.decode(nums)
print("Decoded text:", text_recov)
