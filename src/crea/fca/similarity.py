from __future__ import annotations

import pandas as pd
from concepts import Context


def compute_conceptual_similarity_matrix(context: Context) -> pd.DataFrame:
    """
    Compute the conceptual similarity between objects from the lattice.
    The similarity is:
        sim(i, j) = |C(i) ∩ C(j)| / |C(i) ∪ C(j)|  (Jaccard over concept sets)
    Args:
        context: concepts.Context loaded from a .cxt
    Returns:
        df: Square matrix (objects × objects) with values in [0, 1]
    """

    object_to_concepts = {obj: set() for obj in context.objects}
    for cid, concept in enumerate(context.lattice):
        for obj in concept.extent:
            object_to_concepts[obj].add(cid)

    objects = context.objects
    sim_matrix = pd.DataFrame(index=objects, columns=objects, dtype=float)

    for i in objects:
        ci = object_to_concepts[i]
        for j in objects:
            cj = object_to_concepts[j]
            inter = ci & cj
            union = ci | cj
            sim_matrix.loc[i, j] = (len(inter) / len(union)) if union else 0.0

    return sim_matrix


def compute_mutual_impact_matrix(context: Context) -> pd.DataFrame:
    """
    Compute the mutual impact between attributes and objects from the lattice.

    Let C be the set of lattice concepts. For object o and attribute a:
    C_o = { c ∈ C : o ∈ extent(c) }   # concepts that list o among their objects
    C_a = { c ∈ C : a ∈ intent(c) }   # concepts that list a among their attributes

    The mutual impact is the Jaccard ratio over concept sets:
        impact(a, o) = |C_o ∩ C_a| / |C_o ∪ C_a|
    Args:
      context: concepts.Context loaded from a .cxt
    Returns:
      df: Impact matrix (rows = attributes and columns = objects)
    """
    objects = list(context.objects)
    attributes = list(context.properties)

    obj_to_concepts = {o: set() for o in objects}       # C_o
    attr_to_concepts = {a: set() for a in attributes}   # C_a
    for cid, concept in enumerate(context.lattice):
        for o in concept.extent:
            obj_to_concepts[o].add(cid)
        for a in concept.intent:
            attr_to_concepts[a].add(cid)

    # Impact matrix (rows = attributes, cols = objects)
    impact_df = pd.DataFrame(index=attributes, columns=objects, dtype=float)
    impact_df.index.name = "attribute"
    impact_df.columns.name = "object"

    for a in attributes:
        C_a = attr_to_concepts[a]
        for o in objects:
            C_o = obj_to_concepts[o]
            inter = len(C_a & C_o)
            union = len(C_a | C_o)
            impact_df.loc[a, o] = round(inter / union, 4) if union else 0.0

    return impact_df