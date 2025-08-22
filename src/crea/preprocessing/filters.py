from pathlib import Path
import time
import csv


def _filter_csv_stream(fin, fout, threshold=0.03, coherence_col=None, delimiter=';'):
    """
    Filter a CSV stream by a chosen score column.
    Args:
        fin: Input text stream
        fout: Output text stream
        threshold: Keep rows with values >= threshold
        coherence_col: Column name to use (required)
        delimiter: CSV delimiter
    Returns:
        int: Number of kept rows
    """
    if coherence_col is None:
        raise ValueError("Specify coherence_col (e.g., 'coherenceScore')")

        # sanitize iterator: strip NULs if any

    def _iter_clean_lines(f):
        for line in f:
            if '\x00' in line:
                line = line.replace('\x00', '')
            yield line

    reader = csv.reader(_iter_clean_lines(fin), delimiter=delimiter)
    writer = csv.writer(fout, delimiter=delimiter)

    header = next(reader, None)
    if header is None:
        return 0

    header[0] = header[0].lstrip("\ufeff")  # strip possible BOM
    lookup = {h.strip().lower(): i for i, h in enumerate(header)}
    key = coherence_col.strip().lower()
    if key not in lookup:
        raise ValueError(f"Column '{coherence_col}' not found. Available: {list(lookup.keys())}")
    idx = lookup[key]

    writer.writerow(header)

    kept = 0
    for row in reader:
        if idx >= len(row):
            continue
        try:
            if float(row[idx]) >= threshold:
                writer.writerow(row)
                kept += 1
        except ValueError:
            continue
    return kept


def filter_csv_by_coherence(input_csv, output_csv, threshold=0.03, coherence_col=None, delimiter=';'):
    """
    Filter a single annotations CSV by coherence score.
    Args:
        input_csv: Source CSV path
        output_csv: Destination CSV path
        threshold: Keep rows with values >= threshold
        coherence_col: Column name to use (required)
        delimiter: CSV delimiter
    Returns:
        int: Number of kept rows
    """
    in_path = Path(input_csv)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
        return _filter_csv_stream(
            fin, fout,
            threshold=threshold,
            coherence_col=coherence_col,
            delimiter=delimiter,
        )


def filter_dir_by_coherence(input_dir, output_dir, threshold=0.03, coherence_col=None, delimiter=';'):
    """
    Filter all annotation CSV files in a directory by coherence score.
    Args:
        input_dir: Folder with input CSV files
        output_dir: Folder for filtered CSV files
        threshold: Keep rows with values >= threshold
        coherence_col: Column name to use (required)
        delimiter: CSV delimiter
    Returns:
        dict[str, int]: filename → kept rows
    """
    if coherence_col is None:
        raise ValueError("Specify coherence_col (e.g., 'coherenceScore')")

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for csv_path in sorted(in_dir.glob("*.csv")):
        out_path = out_dir / csv_path.name
        with csv_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
            kept = _filter_csv_stream(
                fin, fout,
                threshold=threshold,
                coherence_col=coherence_col,
                delimiter=delimiter,
            )
            results[csv_path.name] = kept
    return results


if __name__ == "__main__":
    # ───────── Standalone run: annotations filtering ─────────
    # Purpose : Annotation filtering on a file and/or a directory.
    # Output  : Filtered CSVs annotations
    # Usage   : Adjust CONFIG below and run this file directly.
    # ─────────────────────────────────────────────────────────

    # CONFIG
    RUN_FILE_FILTERING = False
    RUN_DIR_FILTERING = True

    THRESHOLD = 0.05
    COHERENCE_COL = "coherence"
    DELIMITER = ';'

    INPUT_CSV = Path(
        "/Users/monicasen/PycharmProjects/topic-modeling/datasets/rychkova_papers/annotations/abstracts/raw/rychkova_abstracts_cleaned_fr.csv")
    OUTPUT_CSV = Path(
        "/Users/monicasen/PycharmProjects/topic-modeling/datasets/rychkova_papers/annotations/abstracts/filtered/rychkova_abstracts_filtered_fr.csv")

    INPUT_DIR = Path(
        "/Users/monicasen/PycharmProjects/topic-modeling/datasets/rychkova_papers/annotations/fulltext/raw")
    OUTPUT_DIR = Path(
        "/Users/monicasen/PycharmProjects/topic-modeling/datasets/rychkova_papers/annotations/abstracts/filtered/c050")


    # Helper: count data rows (excludes header)
    def _count_rows(csv_path: Path, delimiter=DELIMITER) -> int:
        # same NUL-sanitizer
        def _iter_clean_lines(f):
            for line in f:
                yield line.replace('\x00', '')

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(_iter_clean_lines(f), delimiter=delimiter)
            total = sum(1 for _ in reader)
        return max(0, total - 1)


    start = time.perf_counter()

    if RUN_FILE_FILTERING:
        if INPUT_CSV.exists():
            out_name = INPUT_CSV.name.replace("_cleaned_", "_filtered_")
            if out_name == INPUT_CSV.name:
                out_name = f"{INPUT_CSV.stem}_filtered{INPUT_CSV.suffix}"
            out_csv = OUTPUT_CSV.parent / out_name

            out_csv.parent.mkdir(parents=True, exist_ok=True)
            kept = filter_csv_by_coherence(
                input_csv=INPUT_CSV,
                output_csv=out_csv,
                threshold=THRESHOLD,
                coherence_col=COHERENCE_COL,
                delimiter=DELIMITER,
            )
            total = _count_rows(INPUT_CSV, DELIMITER)
            pct = (kept / total * 100.0) if total else 0.0
            print(f"| File: kept {kept}/{total} rows ({pct:.1f}%) with {COHERENCE_COL} ≥ {THRESHOLD} → {out_csv}")
        else:
            print(f"| File skipped: {INPUT_CSV} not found")

    if RUN_DIR_FILTERING:
        if INPUT_DIR.exists():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            total_kept = 0
            total_rows = 0
            processed = 0

            for csv_path in sorted(INPUT_DIR.glob("*.csv")):
                # derive output name by replacing "_cleaned_" → "_filtered_"
                out_name = csv_path.name.replace("_cleaned_", "_filtered_")
                if out_name == csv_path.name:
                    out_name = f"{csv_path.stem}_filtered{csv_path.suffix}"
                out_path = OUTPUT_DIR / out_name

                out_path.parent.mkdir(parents=True, exist_ok=True)
                kept = filter_csv_by_coherence(
                    input_csv=csv_path,
                    output_csv=out_path,
                    threshold=THRESHOLD,
                    coherence_col=COHERENCE_COL,
                    delimiter=DELIMITER,
                )
                rows = _count_rows(csv_path, DELIMITER)
                pct = (kept / rows * 100.0) if rows else 0.0
                print(f"| {csv_path.name}: kept {kept}/{rows} ({pct:.1f}%) → {out_path.name}")

                total_kept += kept
                total_rows += rows
                processed += 1

            overall_pct = (total_kept / total_rows * 100.0) if total_rows else 0.0
            print(
                f"| Dir: processed {processed} files, kept {total_kept}/{total_rows} rows ({overall_pct:.1f}%) → {OUTPUT_DIR}")
        else:
            print(f"| Dir skipped: {INPUT_DIR} not found")

    elapsed = time.perf_counter() - start
    print(f"| Done in {elapsed:.2f}s")
