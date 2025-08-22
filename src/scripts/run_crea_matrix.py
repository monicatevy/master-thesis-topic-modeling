"""
CREA — Matrix build & FCA export

Stage: After filtering Babelfy annotations

What this script does:
1) Build synset dictionary (bnid → representative word)
2) Compute term occurrences per document
3) Build the matrix (docs × terms)
4) Normalize rows to percentages
5) Transpose to (terms × docs) ← objects = terms, attributes = documents
6) Binarize under multiple strategies
7) Export .cxt and render the lattice for each strategy
8) Store all artifacts + write report

Notes:
- Optional outputs controlled by WRITE + maybe()
- Optional rendering controlled by RENDER
"""
from __future__ import annotations

import time
from concepts import Context
from src.crea.fca.concept import export_cxt
from src.crea.matrices.matrix import *

# ──────────────── CONFIG ────────────────
WRITE = True
RENDER = True

# ──────────────── fixed params ────────────────
INPUT_DIR = Path("/Users/monicasen/PycharmProjects/topic-modeling/datasets/rychkova_papers/annotations/abstracts/filtered/c050")
MATRIX_DIR = Path("/Users/monicasen/PycharmProjects/topic-modeling/datasets/rychkova_papers/matrix")

# ──────────────── strategies to test ────────────────
strategies = [
    ("direct", None),
    ("medium", 1.00),
    ("low", 1.00),
    ("high", 0.75),
    ("high", 1.00),
    ("high", 1.25),
    ("high", 1.50),
    ("high", 1.75),
    ("high", 2.00),
]

# ──────────────── helper ────────────────
def _maybe(path: str | Path) -> str | None:
    """Return path as str if WRITE=True, else None (for optional outputs)."""
    return str(path) if WRITE else None

# ──────────────── matrix pipeline ────────────────
def run_matrix_pipeline(INPUT_CSV: Path) -> None:
    """Run CREA matrix for a single CSV"""
    start = time.time()

    # ──────────────── run directory (timestamped + file stem) ────────────────
    RUN_ID = time.strftime("%Y%m%d-%H%M%S")
    file_stem = INPUT_CSV.stem  # ex: rychkova_abstracts_filtered_en
    RUN_DIR = MATRIX_DIR / f"{file_stem}__{RUN_ID}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────── core artifact paths ────────────────
    DICT_CSV         = RUN_DIR / f"dict.csv"          # optional sub-dict
    OCCURRENCES_PATH = RUN_DIR / f"occurrences.csv"   # required downstream
    MATRIX_PATH      = RUN_DIR / f"matrix.csv"        # optional
    NORMALIZED_PATH  = RUN_DIR / f"normalized.csv"    # optional
    NORM_T_PATH      = RUN_DIR / f"normalized_T.csv"  # optional

    # ──────────────── word mapping ────────────────
    entity_dict = build_synset_dict_from_csv(str(INPUT_CSV), _maybe(DICT_CSV))

    """
    if WRITE:
        subset = subset_dict_for_csv(entity_dict, str(INPUT_CSV))
        pd.DataFrame(sorted(subset.items()), columns=["babelSynsetID", "label"]) \
            .to_csv(DICT_CSV, index=False, sep=';', encoding='utf-8')
    """

    # ──────────────── compute occurrences ────────────────
    _ = compute_term_occurrences(str(INPUT_CSV), str(OCCURRENCES_PATH), entity_dict)

    # ──────────────── build and normalize matrix ────────────────
    df_matrix = build_term_matrix(str(OCCURRENCES_PATH), _maybe(MATRIX_PATH))
    df_normalized = normalize_matrix(df_matrix, _maybe(NORMALIZED_PATH))  # shape (docs, terms)

    # ──────────────── transpose matrix ────────────────
    df_normalized_T = transpose_matrix(df_normalized, _maybe(NORM_T_PATH))  # shape (terms, docs)

    # ──────────────── report init ────────────────
    report_lines = [
        f"Run ID: {RUN_ID}",
        f"Input CSV: {INPUT_CSV}",
        f"Artifacts root: {RUN_DIR}",
        f"Matrix shape: {df_matrix.shape[0]} x {df_matrix.shape[1]} (docs × terms)",
        f"Normalized shape: {df_normalized.shape[0]} x {df_normalized.shape[1]} (docs × terms)",
        f"Normalized_T shape: {df_normalized_T.shape[0]} x {df_normalized_T.shape[1]} (terms × docs)",
        "",
    ]

    # ──────────────── run over strategies ────────────────
    for strat_name, beta in strategies:
        print("=" * 60)
        print(f"│ File = {INPUT_CSV.name}")
        print(f"│ Strategy = {strat_name.upper():<7} | β = {beta}")
        print("=" * 60)

        # ──────────────── one folder per strategy ────────────────
        strat_dir = RUN_DIR / f"bin_{strat_name}_{str(beta).replace('.', '')}"
        strat_dir.mkdir(exist_ok=True)

        BINARIZED_PATH = strat_dir / f"bin.csv"  # optional
        CXT_PATH = strat_dir / f"cxt"            # required for Context.fromfile
        LATTICE_PNG = strat_dir / f"lattice.png"         # optional

        # ──────────────── binarize on terms × documents (objects = terms) ────────────────
        df_bin = binarize_matrix(df_normalized_T, _maybe(BINARIZED_PATH), strat_name, beta)

        # ──────────────── export .cxt ────────────────
        export_cxt(df_bin, str(CXT_PATH))

        # ──────────────── build lattice in memory from the .cxt ────────────────
        context = Context.fromfile(str(CXT_PATH))
        lattice = context.lattice

        # ──────────────── render lattice image (optional) ────────────────
        if RENDER:
            lattice.graphviz().render(str(LATTICE_PNG).removesuffix('.png'), format='png')

        # ──────────────── update report ────────────────
        report_lines.append(f"[{strat_name} | beta={beta}]")
        report_lines.append(f"  bin_csv : {BINARIZED_PATH if WRITE else '(skipped)'}")
        report_lines.append(f"  cxt     : {CXT_PATH}")
        report_lines.append(f"  lattice : {LATTICE_PNG if RENDER else '(skipped render)'}")
        report_lines.append("")

        # ──────────────── save report (append as we go) ────────────────
        REPORT_PATH = RUN_DIR / "run_report.txt"
        REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Run report saved to: {REPORT_PATH}")

    end = time.time()
    print(f"⏱️ Done {len(strategies)} strategies for {INPUT_CSV.name} in {end - start:.2f} seconds.\n")

# ──────────────── iterate over all CSVs ────────────────
if __name__ == "__main__":
    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV found in {INPUT_DIR}")

    """
    en_csvs = [p for p in csv_files if "_en_" in p.stem]
    fr_csvs = [p for p in csv_files if "_fr_" in p.stem]

    # Global Artefact
    DICT_MASTER = MATRIX_DIR / "dict_master.csv"

    # Build once
    entity_dict_master = build_unified_dict(
        [str(p) for p in en_csvs],
        [str(p) for p in fr_csvs],
        write_csv=_maybe(DICT_MASTER)
    )
    """

    for csv_path in csv_files:
        #run_matrix_pipeline(csv_path, entity_dict_master)
        run_matrix_pipeline(csv_path)
