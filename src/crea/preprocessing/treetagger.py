import csv
import os
import re
import json
import string
from pathlib import Path
import treetaggerwrapper
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

from src.crea.preprocessing.utils import build_txt_corpus

TAGGER_DIR = "/Users/monicasen/PycharmProjects/topic-modeling/src/crea/preprocessing/treetagger"
STOPWORDS = {
    "en": set(stopwords.words("english")),
    "fr": set(stopwords.words("french"))
}
EXTRA_PUNCTUATION = {'“', '”', '«', '»', '‘', '’', '``', "''", "`", "``"}
PUNCTUATION = set(string.punctuation).union(EXTRA_PUNCTUATION)
TAGGERS = {
    "en": treetaggerwrapper.TreeTagger(TAGLANG="en", TAGDIR=TAGGER_DIR),
    "fr": treetaggerwrapper.TreeTagger(TAGLANG="fr", TAGDIR=TAGGER_DIR)
}
CLASS_FILTER = ['ADV', 'DET:ART', 'DET:POS', 'KON', 'PRO', 'PRO:DEM',
                'PRO:IND', 'PRO:PER', 'PRO:POS', 'PRO:REL', 'PRP:det']

def clean_noise(text: str) -> str:
    """
    Clean noisy symbols and normalize text formatting.
    Args:
        text (str): Raw input text
    Returns:
        str: Cleaned text
    """

    # 01. Noisy symbol repetitions (###, ~~~, ^^^)
    text = re.sub(r'([#~^=_+\-])\1{2,}', '', text)

    # 02. Placeholder-like patterns [...] (...)
    text = re.sub(r'\[\.+\]|\(\.+\)', ' ', text)

    # 03. Long ellipses
    text = re.sub(r'\.{3,}', ' ', text)

    # 04. Normalize spacing
    text = text.replace('\t', ' ')
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 05. Normalize line breaks
    text = re.sub(r'(\s*\n){3,}', '\n\n', text)
    text = re.sub(r'^[ \t\n\r]+', '', text)
    text = re.sub(r' *\n *', '\n', text)

    return text.strip()

def preprocess_text(text: str, lang="en", rm_stopwords=True, lemmatize=True, rm_classfilter=True):
    """
    Preprocess text by tokenizing, cleaning, and optionally removing stopwords/lemmatizing.
    Args:
        text: Input text
        lang: Language code ("EN" or "FR")
        rm_stopwords: Remove stopwords if True
        lemmatize: Apply lemmatization if True
        rm_classfilter: Remove POS classes in CLASS_FILTER if True
    Returns:
        list: List of processed tokens
    """

    regx_punctuation = '|'.join(map(re.escape, PUNCTUATION))
    text = re.sub(f'({regx_punctuation})', r' \1 ', text)
    tokens = word_tokenize(text, language=lang)

    clean_tokens = []
    for token in tokens:
        t = token.lower()
        if t in PUNCTUATION:
            continue
        if t.isdigit():
            continue
        if rm_stopwords and t in STOPWORDS[lang]:
            continue
        clean_tokens.append(t)

    if lemmatize:
        tagger = TAGGERS[lang]
        tagged = treetaggerwrapper.make_tags(tagger.tag_text(clean_tokens), allow_extra=True)

        lemmas = []
        for tag in tagged:
            if hasattr(tag, "what"):  # Skip errors
                continue
            if rm_classfilter and tag.pos in CLASS_FILTER:  # Remove filtered grammatical classes
                continue
            lemmas.append(tag.lemma.split('|')[0])  # Remove disambiguations
        return lemmas

    return clean_tokens

def preprocess_txt_to_json(
        txtdir: str,
        output_json: str,
        lang: str,
        lemmatize: bool = True,
        rm_stopwords: bool = True,
        rm_classfilter: bool = True):
    """
    Build a JSON corpus from a folder of .txt files
    Noise cleaning, stopword removal, POS filtering, and TreeTagger lemmatization.
    Args:
        txtdir: Input directory of .txt files
        output_json: Output JSON path
        lang: Language code ("en", "fr")
        lemmatize: Apply lemmatization
        rm_stopwords: Remove stopwords
        rm_classfilter: Remove POS classes
    Returns:
        None (writes JSON file)
    """

    p = Path(txtdir)
    if not p.is_dir():
        raise ValueError(f"Not a directory: {txtdir}")

    docs = {}
    for fp in tqdm(sorted(p.glob("*.txt")), desc=f"{lang} preprocess"):
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            continue

        text = clean_noise(raw)
        tokens = preprocess_text(
            text=text,
            lang=lang,
            rm_stopwords=rm_stopwords,
            lemmatize=lemmatize,
            rm_classfilter=rm_classfilter
        )
        out_text = " ".join(tokens)
        if out_text:
            docs[fp.stem] = out_text

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"| Saved {len(docs)} docs → {output_json}")

def preprocess_csv_to_json(
    input_csv: str,
    output_json: str,
    lang: str,
    id_col: str = "ID",
    text_col: str = "TEXT",
    lemmatize: bool = True,
    rm_stopwords: bool = True,
    rm_classfilter: bool = True,
) -> None:
    """
    Build a JSON corpus from CSV file
    Noise cleaning, stopword removal, POS filtering, and TreeTagger lemmatization.

    Args:
        input_csv: Path to CSV with columns [ID, TEXT]
        output_json: Output JSON path
        lang: Language code ("en", "fr")
        id_col: Column name for document IDs
        text_col: Column name for document text
        lemmatize: Apply lemmatization
        rm_stopwords: Remove stopwords
        rm_classfilter: Remove POS classes
    Returns:
        None (writes JSON file)
    """

    lang = lang.lower().strip()

    docs: dict[str, str] = {}
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # sanity check on columns
        if reader.fieldnames is None or id_col not in reader.fieldnames or text_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns '{id_col}' and '{text_col}'")

        for row in reader:
            doc_id = (row.get(id_col) or "").strip()
            raw_txt = (row.get(text_col) or "").strip()
            if not doc_id or not raw_txt:
                continue

            cleaned = clean_noise(raw_txt)
            if lemmatize or rm_stopwords or rm_classfilter:
                tokens = preprocess_text(
                    text=cleaned,
                    lang=lang,
                    rm_stopwords=rm_stopwords,
                    lemmatize=lemmatize,
                    rm_classfilter=rm_classfilter,
                )
                out_text = " ".join(tokens).strip()
            else:
                out_text = cleaned

            if out_text:
                docs[doc_id] = out_text  # last wins if duplicate IDs

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # ───────── Standalone run: preprocessing pipeline ───────────────
    # Purpose : Clean TXT/CSV files, apply TreeTagger, and output JSON.
    # Output  : {doc_id: processed_text} → input for Babelfy.
    # Usage   : Run this file directly.
    # ────────────────────────────────────────────────────────────────
    # CONFIG

    LANG = "en"
    DATA_ROOT = "/Users/monicasen/PycharmProjects/topic-modeling/datasets/rychkova_papers/data"

    TXT_DIR = os.path.join(DATA_ROOT, "extracted", LANG)
    OUT_CLEAN = os.path.join(DATA_ROOT, "preprocessed", "fulltext", f"rychkova_clean_{LANG}.json")
    OUT_LEMMA = os.path.join(DATA_ROOT, "preprocessed", "fulltext", f"rychkova_lemmas_{LANG}.json")

    # ──────────────── Build the json once ────────────────
    CORPUS_DIR = os.path.join(DATA_ROOT, "extracted", LANG)
    CORPUS_JSON = os.path.join(DATA_ROOT, "preprocessed", f"rychkova_corpus_{LANG}.json")

    FORCE_REBUILD_CORPUS = False
    if FORCE_REBUILD_CORPUS or not os.path.isfile(CORPUS_JSON):
        os.makedirs(os.path.dirname(CORPUS_JSON), exist_ok=True)
        build_txt_corpus(CORPUS_DIR, CORPUS_JSON)

    # ──────────────── preprocess ────────────────
    preprocess_txt_to_json(
        txtdir=TXT_DIR,
        output_json=OUT_CLEAN,
        lang=LANG,
        lemmatize=False,
        rm_stopwords=True,
        rm_classfilter=True,
    )

    preprocess_txt_to_json(
        txtdir=TXT_DIR,
        output_json=OUT_LEMMA,
        lang=LANG,
        lemmatize=True,
        rm_stopwords=True,
        rm_classfilter=True,
    )