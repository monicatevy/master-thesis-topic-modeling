from __future__ import annotations

from pathlib import Path
import pandas as pd

from concepts import Context


import pandas as pd
from concepts import Context

def lattice_to_table(context: Context) -> pd.DataFrame:
    """
    Convert a lattice to a table.
    Args:
        concepts.Context loaded from a .cxt
    Returns:
      pd.DataFrame indexed by concept_id with columns:
        - extent        (str)
        - intent        (str)
        - n_attributes  (int)
        - n_objects     (int)
    """

    rows = []
    for cid, concept in enumerate(context.lattice):
        attrs = list(concept.extent)
        objs  = list(concept.intent)
        rows.append({
            "concept_id": cid,
            "attributes (extent)": ", ".join(map(str, attrs)),
            "objects (intent)":    ", ".join(map(str, objs)),
            "n_attributes": len(attrs),
            "n_objects":    len(objs),
        })
    df = pd.DataFrame(rows).set_index("concept_id")
    return df


def export_cxt(df: pd.DataFrame, output_cxt: str) -> None:
    """
    Write a Formal Context (.cxt) file compatible with the `concepts` library from a binary matrix.
    Args:
        df: Binary matrix
            rows = objects (e.g., doc_id), columns = attributes (e.g., terms).
        output_cxt: Path to the .cxt output file.
    Returns:
        None (write .cxt file)
    """
    output_path = Path(output_cxt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make values boolean and transpose so that
    # df.columns are objects and df.index are attributes
    df = df.astype(bool).T

    objects = df.columns.tolist()     # original row labels
    attributes = df.index.tolist()    # original column labels

    with output_path.open('w', encoding='utf-8') as f:
        # Header
        f.write("B\n\n")

        # Object and attribute counts
        f.write(f"{len(objects)}\n")
        f.write(f"{len(attributes)}\n\n")

        # Object names
        for obj in objects:
            f.write(f"{obj}\n")

        # Attribute names
        for attr in attributes:
            f.write(f"{attr}\n")

        # One line per object
        # 'X' if the object has that attribute, '.' otherwise
        for obj in objects:
            binary_row = ''.join('X' if val else '.' for val in df[obj])
            f.write(f"{binary_row}\n")

    print("| .cxt file saved")
    print(f"| {len(objects)} objects x {len(attributes)} attributes")

