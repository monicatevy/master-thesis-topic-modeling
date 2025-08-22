from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable
import numpy as np
import pdfplumber
from matplotlib import pyplot as plt

# ── Tokens standards ──
ABSTRACT_TOKENS = {"abstract", "résumé", "summary"}
KW_TOKENS       = {"keywords", "keyword", "motsclés", "Key words"}
REF_TOKENS      = {"references", "bibliography", "références", "bibliographie"}

# ── Normalisation ──
def normalize_token(txt: str) -> str:
    return re.sub(r"[^a-z]", "", txt.lower())

def normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# ── Extraction ──
def extract_words_sorted(page, x_tol: float = 1.0, y_tol: float = 2.0):
    words = page.extract_words(x_tolerance=x_tol, y_tolerance=y_tol) or []
    return sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))

# ── Tables ──
def expand_bbox(b, pad: float = 4.0):
    x0, y0, x1, y1 = map(float, b)
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

def remove_words_in_bboxes(words, bboxes):
    if not bboxes:
        return words
    out = []
    for w in words:
        x0, x1 = float(w["x0"]), float(w["x1"])
        top, bot = float(w["top"]), float(w["bottom"])
        keep = True
        for bx0, by0, bx1, by1 in bboxes:
            if (bx0 <= x0 <= bx1 or bx0 <= x1 <= bx1) and (by0 <= top <= by1 or by0 <= bot <= by1):
                keep = False
                break
        if keep:
            out.append(w)
    return out

def group_by_lines(words, line_tol_px: float | None = None):
    if not words:
        return []
    words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))
    if line_tol_px is None:
        heights = [float(w["bottom"]) - float(w["top"]) for w in words]
        med_h = np.median(heights) if heights else 10.0
        line_tol_px = max(2.0, 0.6 * float(med_h))
    lines, cur, cur_top = [], [], None
    for w in words:
        top = float(w["top"])
        if cur_top is None or abs(top - cur_top) <= line_tol_px:
            cur.append(w)
            cur_top = top if cur_top is None else (0.8 * cur_top + 0.2 * top)
        else:
            lines.append(" ".join(wx["text"] for wx in sorted(cur, key=lambda z: float(z["x0"]))))
            cur, cur_top = [w], top
    if cur:
        lines.append(" ".join(wx["text"] for wx in sorted(cur, key=lambda z: float(z["x0"]))))
    return lines

def join_hyphenated(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if out and out[-1].rstrip().endswith("-"):
            out[-1] = out[-1].rstrip()[:-1] + line.lstrip()
        else:
            out.append(line)
    return out

# ── Detect Abstract / Keywords / References ──
def find_first_token_y(words, token_set: set[str]) -> float | None:
    for w in words:
        if normalize_token(w["text"]) in token_set:
            return float(w["top"])
    return None

def collect_block_bbox(words, start_y: float, stop_y: float | None = None, line_gap: float = 12.0):
    tops = []
    last_top = None
    for w in words:
        y = float(w["top"])
        if y < start_y:
            continue
        if stop_y is not None and y >= stop_y:
            break
        if last_top is None or abs(y - last_top) <= line_gap:
            tops.append(y)
            last_top = y
        else:
            break
    if not tops:
        return start_y, start_y
    return min(tops), max(tops)

def find_last_section_page_top(pdf, ref_tokens: set[str] = REF_TOKENS, x_tol: float = 1.0, y_tol: float = 2.0):
    last_idx, last_top = None, None
    for j, p in enumerate(pdf.pages):
        words_j = extract_words_sorted(p, x_tol, y_tol)
        for w in words_j:
            if normalize_token(w["text"]) in ref_tokens:
                last_idx, last_top = j, float(w["top"])
    return last_idx, last_top

# ── Headers/Footers ──
def detect_repeated_headers_footers(pages_lines: list[list[str]], k: int = 3, min_freq: float = 0.5):
    n_pages = len(pages_lines)
    head_c, foot_c = Counter(), Counter()
    for lines in pages_lines:
        if not lines:
            continue
        heads = [normalize_line(x) for x in lines[:k]]
        foots = [normalize_line(x) for x in lines[-k:]]
        head_c.update(set(heads))
        foot_c.update(set(foots))
    heads_keep = {s for s, c in head_c.items() if c >= min_freq * n_pages and len(s) >= 4}
    foots_keep = {s for s, c in foot_c.items() if c >= min_freq * n_pages and len(s) >= 4}
    return heads_keep, foots_keep

def strip_headers_footers(pages_lines: list[list[str]], heads_norm: set[str], foots_norm: set[str], k: int = 3):
    cleaned = []
    for lines in pages_lines:
        kept = []
        for j, line in enumerate(lines):
            norm = normalize_line(line)
            if j < k and norm in heads_norm:
                continue
            if j >= max(0, len(lines)-k) and norm in foots_norm:
                continue
            kept.append(line)
        cleaned.append(kept)
    return cleaned

# ── References ──
def cut_at_last_section(all_lines: Iterable[str], stop_sections: set[str] = REF_TOKENS):
    ref_re = re.compile(r'^\s*(?:' + "|".join(re.escape(s) for s in stop_sections) + r')\s*$', re.I)
    all_lines = list(all_lines)
    last_idx = max((i for i, ln in enumerate(all_lines) if ref_re.match(ln)), default=None)
    if last_idx is not None:
        return all_lines[:last_idx]
    return all_lines


def infer_split_x(
    words,
    page_width: float,
    *,
    center_band=(0.30, 0.70),
    min_side_ratio=0.25,
    min_words=40,
    bins: int | None = None,
):
    # positions horizontales des mots
    xs = np.array([float(w["x0"]) for w in words if w["text"].strip()])
    if len(xs) < min_words:
        return page_width / 2.0, False, 0.0

    if bins is None:
        bins = max(20, int(np.sqrt(len(xs))))

    # histogramme sur l’intervalle [0, page_width]
    hist, edges = np.histogram(xs, bins=bins, range=(0.0, float(page_width)))
    centers = (edges[:-1] + edges[1:]) / 2.0

    # on cherche la vallée dans la bande centrale (gouttière probable)
    mask = (centers > center_band[0] * page_width) & (centers < center_band[1] * page_width)
    if not mask.any():
        return page_width / 2.0, False, 0.0

    valley_idx_local = np.argmin(hist[mask])
    valley_centers = centers[mask]
    valley_hist     = hist[mask]
    split_x = float(valley_centers[valley_idx_local])
    valley_count = float(valley_hist[valley_idx_local])

    # confiance = pic le plus faible (gauche/droite) rapporté à la vallée
    left_mask  = centers < split_x
    right_mask = centers > split_x
    left_peak  = hist[left_mask].max()  if left_mask.any()  else 0.0
    right_peak = hist[right_mask].max() if right_mask.any() else 0.0
    confidence = (min(left_peak, right_peak) / (valley_count + 1e-9)) if valley_count > 0 else 0.0

    # est-ce bien deux colonnes ? (du texte des deux côtés)
    left_ratio  = float((xs < split_x).mean())
    right_ratio = 1.0 - left_ratio
    is_two = (left_ratio > min_side_ratio) and (right_ratio > min_side_ratio)

    return split_x, bool(is_two), float(confidence)


def extract_one_column_pdf(
    PDF_PATH: str,
    OUTPUT_DIR: str,
    ignore_tables: bool = False,
    ignore_abstract: bool = True,
    ignore_keywords: bool = True,
    ignore_references: bool = True,
    debug: bool = True
) -> None:
    """
    Extract plain text from a one-column scientific paper
    Args:
      PDF_PATH: Path to the input PDF.
      OUTPUT_DIR: Directory to save the .txt file
      ignore_tables: Drop detected table areas
      ignore_abstract: Drop the Abstract block on page 1
      ignore_keywords: Drop the Keywords/Index block
      ignore_references: Drop the References/Bibliography block
      debug: Overlay for visual QA
    Returns:
      None (save the .txt file)
    """
    LINE_GAP = 12.0

    all_text_parts: list[str] = []
    abstract_text = keywords_text = ""
    abstract_top = keywords_top = None
    abstract_ymin = abstract_ymax = None
    keywords_ymin = keywords_ymax = None
    ref_page_idx = ref_top = None

    with pdfplumber.open(PDF_PATH) as pdf:
        if ignore_references:
            ref_page_idx, ref_top = find_last_section_page_top(pdf, REF_TOKENS)

        for i, page in enumerate(pdf.pages):
            raw_words = extract_words_sorted(page, x_tol=1.0, y_tol=2.0)
            words = list(raw_words)

            table_bboxes = []
            if ignore_tables:
                try:
                    table_bboxes = [t.bbox for t in page.find_tables()]
                    words = remove_words_in_bboxes(words, [expand_bbox(b, 4.0) for b in table_bboxes])
                except Exception:
                    pass

            # Page 1 : detect Abstract/Keywords
            if i == 0:
                abstract_top = find_first_token_y(words, {"abstract"})
                keywords_top = find_first_token_y(words, KW_TOKENS)

                if abstract_top is not None:
                    stop_y = keywords_top if keywords_top is not None else None
                    abstract_ymin, abstract_ymax = collect_block_bbox(words, abstract_top, stop_y, line_gap=LINE_GAP)
                    abstract_text = " ".join(
                        w["text"] for w in words
                        if abstract_ymin <= float(w["top"]) <= abstract_ymax
                    )

                if keywords_top is not None:
                    keywords_ymin, keywords_ymax = collect_block_bbox(words, keywords_top, None, line_gap=LINE_GAP)
                    keywords_text = " ".join(
                        w["text"] for w in words
                        if keywords_ymin <= float(w["top"]) <= keywords_ymax
                        and normalize_token(w["text"]) not in KW_TOKENS
                    )

            filtered = []
            for w in words:
                y = float(w["top"])

                if i == 0 and abstract_top is not None and y < abstract_top:
                    continue
                if i == 0 and ignore_abstract and abstract_top is not None:
                    if abstract_ymin is not None and abstract_ymax is not None and abstract_ymin <= y <= abstract_ymax:
                        continue
                if i == 0 and ignore_keywords and keywords_top is not None:
                    if keywords_ymin is not None and keywords_ymax is not None and keywords_ymin <= y <= keywords_ymax:
                        continue
                if ignore_references and ref_page_idx is not None:
                    if i > ref_page_idx:
                        continue
                    if i == ref_page_idx and ref_top is not None and y >= ref_top:
                        continue

                filtered.append(w)

            if debug:
                try:
                    im = page.to_image(resolution=150)

                    grey_rects = [{
                        "x0": float(w["x0"]), "top": float(w["top"]),
                        "x1": float(w["x1"]), "bottom": float(w["bottom"])
                    } for w in raw_words]

                    red_rects = [{
                        "x0": float(w["x0"]), "top": float(w["top"]),
                        "x1": float(w["x1"]), "bottom": float(w["bottom"])
                    } for w in filtered]

                    im.draw_rects(grey_rects, stroke="gray", stroke_width=1)
                    im.draw_rects(red_rects, stroke="red", stroke_width=1)

                    if i == 0:
                        if abstract_ymin is not None and abstract_ymax is not None:
                            im.draw_hline(abstract_ymin, stroke="green", stroke_width=2)
                            im.draw_hline(abstract_ymax, stroke="green", stroke_width=2)
                        if keywords_ymin is not None and keywords_ymax is not None:
                            im.draw_hline(keywords_ymin, stroke="orange", stroke_width=2)
                            im.draw_hline(keywords_ymax, stroke="orange", stroke_width=2)

                    if ignore_references and ref_page_idx is not None and i == ref_page_idx and ref_top is not None:
                        im.draw_hline(ref_top, stroke="purple", stroke_width=2)

                    import matplotlib.pyplot as plt  # assure-toi d'avoir importé plt au top si tu préfères
                    plt.figure(figsize=(8, 10))
                    plt.imshow(im.annotated)
                    plt.axis("off")
                    plt.title(f"Page {i + 1} — gray: all | red: kept")
                    plt.show()
                except Exception as e:
                    print(f"[debug-warning] overlay failed on page {i + 1}: {e}")

            all_text_parts.extend(w["text"] for w in filtered)

    # Assembly
    prefix_parts = []
    if not ignore_abstract and abstract_text:
        prefix_parts.append(abstract_text)
    if not ignore_keywords and keywords_text:
        prefix_parts.append(keywords_text)

    final_text = "\n\n".join(prefix_parts + [" ".join(all_text_parts)])

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / (Path(PDF_PATH).stem + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    return None


def extract_two_column_pdf(
    PDF_PATH: str,
    OUTPUT_DIR: str,
    ignore_tables: bool = False,
    debug: bool = False,
    *,
    force_split_ratio: float | None = None,   # ex: 0.58 (prioritaire sur l'estimation auto)
    force_split_x_px: float | None = None,    # ex: 315.0 (position absolue en px)
    nudge_split_ratio: float = 0.0,           # ex: +0.02 (décale la ligne après estimation)
) -> None:
    """
    Extract plain text from a (possibly) two-column scientific paper and save as .txt.
    Args:
      PDF_PATH: Path to the input PDF
      OUTPUT_DIR: Directory where to write the .txt
      ignore_tables: Drop detected table areas
      debug: Interactive overlay for visual QA (no files written)
      force_split_ratio: Force a fixed split as a ratio of page width (0..1)
      force_split_x_px: Force a fixed split in absolute pixels from the left edge
      nudge_split_ratio: Additive shift applied to the global ratio (+/-)
    Returns:
      None (saves the .txt file)
    """
    PDF_PATH = Path(PDF_PATH)
    OUTPUT_DIR = Path(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ───────── Pass 1: estimer un split global (sauf si on le force) ─────────
    global_ratio = None
    CONF_TAU = 2.0
    MIN_WORDS = 60

    if force_split_ratio is not None:
        # on verrouille directement le ratio global
        global_ratio = float(force_split_ratio)

    if global_ratio is None:
        with pdfplumber.open(str(PDF_PATH)) as pdf:
            ratios = []
            n_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                raw_words = extract_words_sorted(page, x_tol=1.0, y_tol=2.0)
                words = list(raw_words)

                if ignore_tables:
                    try:
                        table_bboxes = [expand_bbox(t.bbox, 4.0) for t in page.find_tables()]
                        words = remove_words_in_bboxes(words, table_bboxes)
                    except Exception:
                        pass

                # stabilise l'estimation : commence après Abstract en page 1 si présent
                if i == 0:
                    start_top = find_first_token_y(raw_words, ABSTRACT_TOKENS)
                    if start_top is not None:
                        thr = float(start_top) - 2.0
                        words = [w for w in words if float(w["top"]) >= thr]

                if len(words) < MIN_WORDS:
                    continue
                if i == 0 or i >= n_pages - 2:
                    continue

                split_x, is_two, conf = infer_split_x(words, float(page.width))
                if is_two and conf >= CONF_TAU:
                    ratios.append(split_x / float(page.width))

            if len(ratios) >= 3:
                global_ratio = float(np.median(ratios))

    # applique le nudge (qu’on ait forcé ou estimé)
    if global_ratio is not None:
        global_ratio = float(global_ratio) + float(nudge_split_ratio)
        # garde-fous
        global_ratio = min(0.85, max(0.15, global_ratio))

    # ───────── Pass 2: extraction avec split fixé (pixel/ratio) ou inféré ─────────
    pages_lines: list[list[str]] = []

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_words = extract_words_sorted(page, x_tol=1.0, y_tol=2.0)
            words = list(raw_words)

            if ignore_tables:
                try:
                    table_bboxes = [expand_bbox(t.bbox, 4.0) for t in page.find_tables()]
                    words = remove_words_in_bboxes(words, table_bboxes)
                except Exception:
                    pass

            # Page 1 : commencer après "Abstract"/"Résumé" si présent
            if i == 0:
                start_top = find_first_token_y(raw_words, ABSTRACT_TOKENS)
                if start_top is not None:
                    thr = float(start_top) - 2.0
                    words = [w for w in words if float(w["top"]) >= thr]

            # --- compute split_x for this page ---
            if force_split_x_px is not None:
                split_x = float(force_split_x_px)
                left_words  = [w for w in words if float(w["x0"]) <  split_x]
                right_words = [w for w in words if float(w["x0"]) >= split_x]
                total = max(1, len(words))
                is_two = (len(left_words) / total >= 0.10) and (len(right_words) / total >= 0.10)

            elif global_ratio is not None:
                split_x = float(page.width) * float(global_ratio)
                left_words  = [w for w in words if float(w["x0"]) <  split_x]
                right_words = [w for w in words if float(w["x0"]) >= split_x]
                total = max(1, len(words))
                is_two = (len(left_words) / total >= 0.10) and (len(right_words) / total >= 0.10)

            else:
                # fallback: inférence page-level
                split_x, is_two, _ = infer_split_x(words, float(page.width))
                left_words  = [w for w in words if float(w["x0"]) <  split_x]
                right_words = [w for w in words if float(w["x0"]) >= split_x]

            # Build lines
            if is_two:
                left_lines  = group_by_lines(left_words)
                right_lines = group_by_lines(right_words)
                lines = join_hyphenated(left_lines) + join_hyphenated(right_lines)
            else:
                lines = join_hyphenated(group_by_lines(words))

            pages_lines.append(lines)

            # Debug overlay — interactif, et on trace la ligne bleue quoi qu'il arrive
            if debug:
                try:
                    im = page.to_image(resolution=150)
                    grey_rects = [
                        {"x0": float(w["x0"]), "top": float(w["top"]), "x1": float(w["x1"]), "bottom": float(w["bottom"])}
                        for w in raw_words
                    ]
                    red_rects = [
                        {"x0": float(w["x0"]), "top": float(w["top"]), "x1": float(w["x1"]), "bottom": float(w["bottom"])}
                        for w in words
                    ]
                    im.draw_rects(grey_rects, stroke="gray", stroke_width=1)
                    im.draw_rects(red_rects,  stroke="red",  stroke_width=1)
                    im.draw_vline(split_x, stroke="blue", stroke_width=2)

                    if i == 0:
                        # petit log utile pour ajuster à la main
                        print(f"[debug] page width = {float(page.width):.1f}px | split_x = {split_x:.1f}px "
                              f"({split_x/float(page.width):.3f} ratio) | is_two={is_two}")

                    plt.figure(figsize=(8, 10))
                    plt.imshow(im.annotated)
                    plt.axis("off")
                    mode = "two-col" if is_two else "one-col"
                    plt.title(f"Page {i+1} — gray: all | red: kept | {mode}")
                    plt.show()
                except Exception as e:
                    print(f"[debug-warning] overlay failed on page {i+1}: {e}")

    # Headers/footers récurrents + coupe à "References"
    heads_norm, foots_norm = detect_repeated_headers_footers(pages_lines, k=3, min_freq=0.5)
    cleaned_pages = strip_headers_footers(pages_lines, heads_norm, foots_norm, k=3)
    all_lines = [ln for page in cleaned_pages for ln in page]
    all_lines = cut_at_last_section(all_lines, REF_TOKENS)

    text = "\n".join(all_lines)

    out_path = OUTPUT_DIR / (PDF_PATH.stem + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return None