import os
import csv
import sys
import tiktoken  # For OpenAI models
from transformers import AutoTokenizer  # For LLaMA models

TOKENIZERS = {
    "gpt-4": tiktoken.encoding_for_model("gpt-4"),
    "llama-3.2-1B": AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B"),
}

if len(sys.argv) < 2:
    print("Usage: python count_tokens.py <directory_path> [output_file]")
    sys.exit(1)

directory_path = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else "token_counts.csv"

results = {}

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    if os.path.isfile(file_path) and filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        results[filename] = {}

        for model, tokenizer in TOKENIZERS.items():
            if model.startswith("gpt"):
                num_tokens = len(tokenizer.encode(content))
            else:  # LLaMA models
                num_tokens = len(tokenizer.encode(content, add_special_tokens=True))

            results[filename][model] = num_tokens

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["Filename"] + list(TOKENIZERS.keys()))

    for filename, token_counts in results.items():
        writer.writerow([filename] + [token_counts[model] for model in TOKENIZERS.keys()])

print(f"✅ Saved to {output_file}")
