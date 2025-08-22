import argparse
import csv
import re
import json
from pathlib import Path
from tqdm import tqdm
from itertools import islice
from babelpy.babelfy import BabelfyClient

from src.crea.preprocessing.utils import distribute_length, load_done_meta, save_done_meta

def annotate_text(doc_id, text, writer, babel_client, base_pack=8000, threshold=5):
    """
    Annotate a document with Babelfy, handling API size limits.
    Args:
        doc_id: Document identifier
        text: Document content
        writer: CSV writer to save annotations
        babel_client: Babelfy client instance
        base_pack: Initial max chunk size
        threshold: Allowed deviation percentage for chunk distribution
    Returns:
        int: Final chunk size used (max_pack accepted by Babelfy)
    Raises:
        RuntimeError: If the document cannot be annotated even at 3000 chars
    """

    current_pack = base_pack
    while current_pack >= 3000:   # plancher de sécurité
        try:
            slices = distribute_length(len(text), max_pack=current_pack, threshold=threshold)
            offset = 0

            for size in slices:
                chunk = text[offset: offset + size]
                babel_client.babelfy(chunk)
                anns = babel_client.merged_entities

                if not anns:
                    writer.writerow([doc_id, '', '', '', '', '0', '0', '0'])
                    offset += size
                    continue

                for ann in anns:
                    frag = ann.get('charFragment', {})
                    start, end = frag.get('start'), frag.get('end')
                    if start is not None and end is not None:
                        start_full = start + offset
                        end_full = end + offset
                        snippet = text[start_full:end_full + 1].replace('\n', ' ').strip()
                    else:
                        start_full = end_full = ''
                        snippet = ''
                    writer.writerow([
                        doc_id, snippet, start_full, end_full,
                        ann.get('babelSynsetID', ''),
                        ann.get('score', 0.0),
                        ann.get('coherenceScore', 0.0),
                        ann.get('globalScore', 0.0)
                    ])
                offset += size

            # si on est arrivé ici → succès avec current_pack
            return current_pack

        except Exception as e:
            if "414" in str(e) or "400" in str(e):
                print(f"| {doc_id}: pack {current_pack} too big → retry with {current_pack-500}")
                current_pack -= 500
            else:
                raise e

    raise RuntimeError(f"Impossible to annotate {doc_id} even at 3000 chars per chunk")


def main() -> None:
    # ────────────── CLI ──────────────
    p = argparse.ArgumentParser()
    p.add_argument('--input-type', choices=['json', 'csv'], default='json',
                   help="Input format: json (dict id->text) or csv (ID,Abstract)")
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--done', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--limit', type=int, default=10_000)
    p.add_argument('--doc-ids', help="Comma-separated doc ids to annotate")
    p.add_argument('--doc-ids-file', help="Path to a file listing doc ids (comma/line separated)")
    p.add_argument('--lang', default='EN')
    p.add_argument('--base-pack', type=int, default=8000)
    p.add_argument('--threshold', type=int, default=5)
    args = p.parse_args()

    # Normalize language once
    lang = args.lang.upper()

    # Create Babelfy instance (use merged_entities)
    babel_client = BabelfyClient(args.key, {'lang': lang})

    # ────────────── load data ──────────────
    if args.input_type == 'json':
        with open(args.input, encoding='utf-8') as f:
            docs = json.load(f)

    elif args.input_type == 'csv':
        with open(args.input, newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            expected = {'ID', 'Abstract'}
            if not expected.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must have headers: {expected}, got {reader.fieldnames}")
            docs = {row['ID']: row['Abstract'] for row in reader if row.get('ID')}

    else:
        raise ValueError(f"Unsupported input-type: {args.input_type}")

    total_docs = len(docs)
    done_docs, meta_info = load_done_meta(args.done)

    # ────────────── optional filter on a subset of docs ──────────────
    wanted_ids: set[str] = set()

    # IDs pass in CLI
    if args.doc_ids:
        wanted_ids |= {x.strip() for x in args.doc_ids.split(',') if x.strip()}

    # IDs read through txt file
    if args.doc_ids_file:
        try:
            with open(args.doc_ids_file, encoding='utf-8') as fh:
                raw = fh.read()
                for token in re.split(r'[,\n]+', raw):
                    token = token.strip()
                    if token:
                        wanted_ids.add(token)
        except OSError as e:
            print(f"| Error reading --doc-ids-file: {e} – exiting.")
            return

    # no filter
    if not wanted_ids:
        wanted_ids = set(docs.keys())

    already_done = wanted_ids & done_docs
    remaining_ids = wanted_ids - done_docs

    if not remaining_ids:
        print("=" * 60)
        print(f"| Requested docs ({len(already_done)}) already annotated.")
        print("=" * 60)
        return

    docs_to_process = {k: docs[k] for k in remaining_ids if k in docs}

    # ────────────── launch infos ──────────────
    print("=" * 60)
    print(f"| Start: {Path(args.input).name}")
    print(f"| Already annotated: {len(done_docs)} (skipped)")
    print(f"| Remaining: {len(docs_to_process)} / Total: {total_docs}")
    print("=" * 60)

    # ────────────── output folders ──────────────
    for folder in (Path(args.output).parent,
                   Path(args.done).parent):
        folder.mkdir(parents=True, exist_ok=True)

    # ────────────── main loop ──────────────
    nb_annotated = 0

    with open(args.output,  'a', newline='', encoding='utf-8') as csvfile:

        writer = csv.writer(csvfile, delimiter=';')
        if csvfile.tell() == 0:                 # header seulement si nouveau fichier
            writer.writerow(['doc_id','word','start','end',
                             'babelSynsetID','score','coherence','global'])

        todo = min(args.limit, len(docs_to_process))
        stream = islice(docs_to_process.items(), todo)

        for doc_id, text in tqdm(stream, total=todo, desc="Progress"):
            try:
                used_pack = annotate_text(doc_id, text, writer, babel_client,
                                          base_pack=args.base_pack, threshold=args.threshold)
                done_docs.add(doc_id)
                meta_info[doc_id] = {"max_pack": used_pack}
                nb_annotated += 1
            except Exception as e:
                print(f"| Error on {doc_id}: {e}")

    # ────────────── save annotated docs ──────────────
    save_done_meta(args.done, done_docs, meta_info)


    # ────────────── end ──────────────
    print(f"| Finished {nb_annotated} annotations")
    print(f"| Output saved to: {args.output}")
    print("="*60)

if __name__ == '__main__':
    main()