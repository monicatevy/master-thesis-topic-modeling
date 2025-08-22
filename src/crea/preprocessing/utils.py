import json
import os
from pathlib import Path

def build_txt_corpus(txtdir: str, output_json: str):
    """
    Convert a folder of .txt files into a JSON corpus.
    Each file becomes one document:
      - doc_id = filename
      - text   = file content (UTF-8)
    Args:
        txtdir: Directory containing .txt files
        output_json: Path to save {doc_id: text} mapping
    Returns:
        None (writes JSON file)
    """
    p = Path(txtdir)
    if not p.is_dir():
        raise ValueError("Input path must be a directory containing .txt files")

    docs = {}
    for fp in sorted(p.glob("*.txt")):
        doc_id = fp.stem
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs[doc_id] = text

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"| Corpus built : {len(docs)} docs saved → {output_json}")

def distribute_length(number, max_pack=8000, threshold=20):
    """
    Split a number into chunks of size close to `maxPack`, allowing variation
    within a percentage threshold.
    Args:
        number: Total value to distribute
        max_pack: Target packet size (>0)
        threshold: Allowed deviation in percentage (0–100)
    Returns:
        list: List of packet sizes
    """

    if max_pack <= 0:
        print("maxPack must be positive AND above 0")
        return []

    if threshold < 0 or threshold > 100:
        print("Threshold is a percentage (20% ideally) between 0 and 100")
        return []

    if number < max_pack:
        return [number]

    q, r = number // max_pack, number % max_pack

    if r == 0:
        return [max_pack] * q

    maxT = max_pack + ((max_pack * threshold) // 100)
    minT = max_pack - ((max_pack * threshold) // 100)

    # Case 1: remainder is large enough -> keep it as a separate chunk
    if r > maxT:
        L = [max_pack] * q
        L.append(r)
        return L

    q2, r2 = r // q, r % q

    # Case 2: remainder can be evenly distributed among chunks
    if max_pack + q2 + r2 < maxT:
        L = [max_pack + q2] * q
        L[-1] += r2
        return L

    # Case 3: remainder too small/large -> retry with a lower target size
    return distribute_length(number, minT, threshold)

def load_done_meta(path: str):
    """
    Load tracking metadata of already processed documents.
    Args:
        path: Path to the JSON metadata file
    Returns:
        tuple:
        - set of processed document IDs
        - dict mapping doc_id → {"max_pack": int}
    """

    if not os.path.exists(path):
        return set(), {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data), {}
    if isinstance(data, dict):
        return set(data.get("done_ids", [])), (data.get("info") or {})
    return set(), {}

def save_done_meta(path: str, done_ids: set[str], info: dict):
    """
    Save tracking metadata of processed documents.
    Args:
        path: Output JSON path
        done_ids: Set of processed document IDs
        info: Dict mapping doc_id → {"max_pack": int}
    Returns:
        None (writes JSON file)
    """

    payload = {"done_ids": sorted(done_ids), "info": info}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)