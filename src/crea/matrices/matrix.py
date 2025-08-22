from __future__ import annotations

import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict


# helpers
def _format_shape(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    idx = (df.index.name or "").lower()
    col = (df.columns.name or "").lower()
    if idx == "doc_id" or col == "term":
        return f"{rows} documents x {cols} terms"
    if col == "doc_id" or idx == "term":
        return f"{cols} documents x {rows} terms"
    return f"{rows} rows x {cols} columns"

def build_synset_dict_from_csv(input_csv: str, output_path: str | None = None) -> dict[str, str]:
    """
    Build a synset lexicon (synset_id → representative word) from a Babelfy-annotated CSV.
    Args:
        input_csv: Path to the babelfy annotations CSV
        output_path: Optional path to write the full lexicon CSV
    Returns:
        dict[str, str]: Maps each babelSynsetID to one representative word,
        selected as the first word in alphabetical order.
    """
    bnid_to_words = defaultdict(set)

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            bnid = row.get('babelSynsetID', '').strip()
            word = row.get('word', '').strip()
            if bnid and word:
                bnid_to_words[bnid].add(word)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['babelSynsetID', 'words'])
            for bnid, words in sorted(bnid_to_words.items()):
                writer.writerow([bnid, ', '.join(sorted(words))])
        print("=" * 60)
        print(f"| Synset-to-word dictionary saved to: {out}")
    else:
        print("=" * 60)
        print("| Synset-to-word dictionary built (not saved)")

    print(f"| Total unique babelSynsetIDs: {len(bnid_to_words)}")

    return {bnid: sorted(words)[0] for bnid, words in bnid_to_words.items()}


def compute_term_occurrences(input_csv: str, output_csv: str | None, entity_dict: dict[str, str]) -> pd.DataFrame:
    """
    Count per-document occurrences of babelSynsetID
    Args:
        input_csv: Path to the babelfy annotations CSV
        output_csv: Optional path to write the occurrences CSV
        entity_dict: Mapping {babelSynsetID -> representative word}
    Returns:
        df: Table with columns ['doc_id', 'babelSynsetID', 'word', 'count'].
    """

    df = pd.read_csv(input_csv, sep=';')

    df = df[df['babelSynsetID'].notnull() & (df['babelSynsetID'] != '')]
    df['word'] = df['babelSynsetID'].map(entity_dict).fillna("UNKNOWN")

    # Group occurrences
    grouped = (
        df.groupby(['doc_id', 'babelSynsetID', 'word'])
          .size()
          .reset_index(name='count')
    )

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(output_path, index=False, sep=';')
        print(f"| Occurrence table saved to {output_path}")
    else:
        print("| Occurrence table built")

    return grouped


def build_term_matrix(input_csv: str, output_csv: str | None = None) -> pd.DataFrame:
    """
    Aggregate (doc_id, word, count) into a wide document–term matrix.
    Args:
        input_csv: Path to the CSV (as produced by `compute_term_occurrences`).
        output_csv: Optional path to write the matrix
    Returns:
        df: Dense matrix of shape (docs, terms) with integer counts.
    """

    input_path = Path(input_csv)
    df = pd.read_csv(input_path, sep=';')

    # Group and pivot to a matrix
    matrix = df.groupby(['doc_id', 'word'])['count'].sum().unstack(fill_value=0)

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(output_path, sep=';')

    print("-" * 60)
    print(f"| Matrix built")
    print(f"| Shape: {_format_shape(matrix)}")

    return matrix


def normalize_matrix(df: pd.DataFrame, output_csv: str | None = None) -> pd.DataFrame:
    """
    Normalize each document row to percentages (row sum ≈ 100).
    Args:
        df: Document–term matrix.
        output_csv: Optional path to write the normalized matrix.
    Returns:
        df: Shape (docs, terms), floats.
    """
    normalized_df = df.div(df.sum(axis=1), axis=0).fillna(0) * 100

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        normalized_df.to_csv(out, sep=';')

    rows, cols = normalized_df.shape
    print("-" * 60)
    print("| Normalized matrix")
    print(f"| Shape: {_format_shape(normalized_df)}")

    return normalized_df


def transpose_matrix(df: pd.DataFrame, output_csv: str | None = None) -> pd.DataFrame:
    """
    Transpose a matrix
    From shape (docs, terms) → shape (terms, docs)
    Args:
        df: Matrix
        output_csv: Optional path to write the transposed matrix
    Returns:
        df: Transposed matrix
    """
    transposed = df.transpose()

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        transposed.to_csv(out, sep=';')

    print("-" * 60)
    print("| Transposed matrix")
    print(f"| Shape: {_format_shape(transposed)}")

    return transposed


def binarize_matrix(
    df: pd.DataFrame,
    output_csv: str | None = None,
    strategy: str = "high",
    beta: float = 0.5
) -> pd.DataFrame:
    """
    Binarize a document–term matrix using strategies.
    Args:
        df: Matrix
        output_csv: Optional path to write the binarized matrix
        strategy:
            - "direct": 1 if df > 0 (no thresholding).
            - "high":   threshold = mean + beta * std   (row-wise, on frequencies).
            - "medium": threshold = mean                (row-wise, on frequencies).
            - "low":    threshold = mean - beta * std   (row-wise, on frequencies).
        beta: Scaling factor for "high" and "low" strategies.
    Returns:
        df: Binary matrix
    """

    if strategy not in {"direct", "high", "medium", "low"}:
        raise ValueError("Unknown strategy. Choose from 'direct', 'high', 'medium', 'low'.")

    if strategy == "direct":
        binary = (df > 0).astype(int)
    else:
        # Normalize each row to relative frequencies
        row_sum = df.sum(axis=1)
        freq = df.div(row_sum, axis=0).fillna(0)

        if strategy == "high":
            threshold = freq.mean(axis=1) + beta * freq.std(axis=1)
        elif strategy == "medium":
            threshold = freq.mean(axis=1)
        else:  # "low"
            threshold = freq.mean(axis=1) - beta * freq.std(axis=1)

        binary = freq.gt(threshold, axis=0).astype(int)

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        binary.to_csv(out, sep=';')
        print("| Binarized matrix saved to", out)
    else:
        print("| Binarized matrix built")

    print(f"| Strategy: {strategy}" + (f" (beta = {beta})" if strategy != "direct" else ""))
    print(f"| Shape: {_format_shape(binary)}")

    return binary