import csv
import os
import shutil
import sys

from src.crea.preprocessing.treetagger import distribute_length

sys.path.append('')
import json
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 7500
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def load_documents(source_path: str, source_type: str = "json", filter_ids: set = None, filenames: list = None):
    """
    Load documents from a JSON, TXT folder, or CSV file (with 'ID' and 'Abstract' columns).
    Optionally filter by a whitelist of IDs or filenames.
    """
    documents = []
    base_name = ""

    if source_type == "txt":
        filenames = sorted(f for f in os.listdir(source_path) if f.endswith(".txt")) if filenames is None else filenames
        for fname in filenames:
            if filter_ids and fname not in filter_ids:
                continue
            with open(os.path.join(source_path, fname), "r", encoding="utf-8") as f:
                documents.append((fname, f.read().strip()))
        folder = os.path.basename(source_path.rstrip("/\\"))
        base_name = f"{folder}_{len(documents)}"

    elif source_type == "json":
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if not filter_ids or k in filter_ids:
                documents.append((k, v.strip()))
        base_name = os.path.splitext(os.path.basename(source_path))[0]

    elif source_type == "csv":
        with open(source_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get('ID')
                text = row.get('TEXT')
                if doc_id and text and (not filter_ids or doc_id in filter_ids):
                    documents.append((doc_id, text.strip()))
        base_name = os.path.splitext(os.path.basename(source_path))[0]

    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    return documents, base_name


def build_prompts(input_path: str, output_dir: str,
                  prompt_header: str, prompt_footer: str,
                  source_type: str = "json",
                  w_chunk_id: bool = True,
                  filenames: list = None,
                  filter_ids: set = None,
                  filename_prefix: str = None) -> list[str]:

    prompts = []
    current_prompt = prompt_header
    current_tokens = len(tokenizer.tokenize(prompt_header + prompt_footer))
    has_content = False
    prompt_idx = 1

    os.makedirs(output_dir, exist_ok=True)

    docs, base_name = load_documents(input_path, source_type=source_type, filter_ids=filter_ids, filenames=filenames)
    prefix = filename_prefix or f"{base_name}_prompt"

    def flush_prompt(idx):
        nonlocal current_prompt, current_tokens, has_content
        prompt = current_prompt + prompt_footer
        if has_content:
            prompts.append(prompt)

            # Save
            filename = f"{prefix}_prompt_{idx:02d}.txt"
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"Saved {filename}")

        # Reset
        current_prompt = prompt_header
        current_tokens = len(tokenizer.tokenize(prompt_header + prompt_footer))
        has_content = False

    # Loop through documents
    for doc, content in docs:
        tokens = tokenizer.tokenize(content)
        num_tokens = len(tokens)

        if num_tokens <= MAX_TOKENS:
            doc_text = tokenizer.convert_tokens_to_string(tokens)
            cleaned_text = doc_text.strip().replace("\n", " ")
            block = (
                f'    <document>\n'
                f'      {cleaned_text}\n'
                f'    </document>\n'
            )
            block_tokens = len(tokenizer.tokenize(block))
            if current_tokens + block_tokens > MAX_TOKENS:
                flush_prompt(prompt_idx)
                prompt_idx += 1
            current_prompt += block
            current_tokens += block_tokens
            has_content = True
            print(f"{doc} | {num_tokens} tokens")

        else:
            chunk_sizes = distribute_length(num_tokens, max_pack=MAX_TOKENS)
            print(f"{doc} | {num_tokens} tokens → chunks: {chunk_sizes}")
            start = 0

            for i, size in enumerate(chunk_sizes, 1):
                chunk_tokens = tokens[start:start + size]
                chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
                chunk_text_clean = chunk_text.strip().replace("\n", " ")

                if w_chunk_id:
                    chunk_block = (
                        f'    <document>\n'
                        f'      <chunk i="{i}">\n'
                        f'        {chunk_text_clean}\n'
                        f'      </chunk>\n'
                        f'    </document>\n'
                    )
                else:
                    chunk_block = (
                        f'    <document>\n'
                        f'      <chunk>\n'
                        f'        {chunk_text_clean}\n'
                        f'      </chunk>\n'
                        f'    </document>\n'
                    )

                block_tokens = len(tokenizer.tokenize(chunk_block))
                if current_tokens + block_tokens > MAX_TOKENS:
                    flush_prompt(prompt_idx)
                    prompt_idx += 1

                current_prompt += chunk_block
                current_tokens += block_tokens
                has_content = True
                start += size

    # Last flush
    if current_prompt != prompt_header:
        flush_prompt(prompt_idx)

    return prompts


def export_prompt_config(prompt_dir: str, output_zip_dir: str, zip_name: str) -> str:
    """
    Crée une archive ZIP contenant les prompts.

    :param prompt_dir: dossier contenant les .txt générés
    :param output_zip_dir: dossier où sauvegarder le .zip
    :param zip_name: nom voulu pour l'archive (sans .zip)
    :return: chemin complet du fichier zip créé
    """
    os.makedirs(output_zip_dir, exist_ok=True)
    zip_path = os.path.join(output_zip_dir, zip_name)
    shutil.make_archive(zip_path, 'zip', prompt_dir)

    print(f"✅ ZIP created: {zip_path}.zip")
    return zip_path + ".zip"
