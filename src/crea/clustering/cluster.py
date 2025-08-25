from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

def compute_silhouette_for_k(
    sim_matrix: pd.DataFrame,
    Z: np.ndarray,
    k: int,
) -> tuple[float, pd.Series]:
    """
    Compute the mean Silhouette score for a given number of clusters k,
    using a precomputed hierarchical linkage (Z) and a similarity matrix.
    Args:
        sim_matrix: Square DataFrame of similarities in [0,1]
        Z: Z: SciPy linkage matrix
        k: Number of clusters to form
    Returns:
        A tuple (silhouette_mean, labels) where:
          - silhouette_mean: Mean Silhouette score for the partition at k.
          - labels: Series indexed by terms (sim_matrix.index) with integer cluster IDs in [1..k].
    """

    # Build distance matrix
    similarity = sim_matrix.astype(float).copy()
    dist = 1.0 - similarity
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist.values, 0.0)

    # Flat clustering
    labels_arr = fcluster(Z, t=int(k), criterion="maxclust")
    labels = pd.Series(labels_arr.astype(int), index=sim_matrix.index, name="cluster")

    # At least 2 distinct labels and < n_samples clusters
    unique_labels = np.unique(labels_arr)
    if unique_labels.size < 2 or unique_labels.size >= dist.shape[0]:
        return float("nan"), labels

    # Compute silhouette
    try:
        silhouette = float(silhouette_score(dist.values, labels_arr, metric="precomputed"))
    except Exception:
        silhouette = float("nan")

    return silhouette, labels


def compute_hierarchical_clusters(
    sim_matrix: pd.DataFrame,
    threshold: float | None = None,
    n_clusters: int | None = None,
    linkage_method: str = "complete",
):
    """
    Compute hierarchical clustering from a similarity matrix and derive flat clusters.
    Args:
      sim_matrix: Square DataFrame of similarities in [0,1]
      threshold: Cut distance used to derive flat clusters (criterion="distance")
      n_clusters: Desired number of clusters (criterion="maxclust")
      linkage_method: SciPy linkage method (e.g., 'average', 'complete', 'ward')
    Returns:
      Z: SciPy linkage matrix
      clusters: pd.Series of cluster labels indexed by object
      labels: list of object labels (row/col index of sim_matrix)
    """

    if (threshold is not None) and (n_clusters is not None):
        raise ValueError("Provide either 'threshold' or 'n_clusters', not both.")
    if not isinstance(sim_matrix, pd.DataFrame) or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("sim_matrix must be a square pandas.DataFrame.")
    if sim_matrix.isna().any().any():
        raise ValueError("sim_matrix contains NaNs; please clean or impute before clustering.")
    if (sim_matrix.values < 0).any() or (sim_matrix.values > 1).any():
        raise ValueError("Similarities must be in [0, 1].")

    # Build distance matrix
    similarity = sim_matrix.astype(float).copy()
    dist = 1.0 - similarity
    labels = dist.index.tolist()
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist.values, 0.0)

    # Condensed vector for SciPy
    dist_condensed = squareform(dist.values, checks=False)
    Z = linkage(dist_condensed, method=linkage_method)

    clusters = None
    if threshold is not None:
        clusters = pd.Series(
            fcluster(Z, t=threshold, criterion="distance"),
            index=labels,
            name="cluster",
        )
    elif n_clusters is not None:
        clusters = pd.Series(
            fcluster(Z, t=n_clusters, criterion="maxclust"),
            index=labels,
            name="cluster",
        )
    return Z, clusters, labels


def clusters_to_df(clusters: pd.Series, save_path: str | None = None) -> pd.DataFrame:
    """
    Build a cluster table from cluster labels.
    Args:
      clusters: pd.Series mapping each term (index) to a cluster id (values)
      save_path: Optional path to save the CSV
    Returns:
      df: Columns = ['cluster_id', 'n_terms', 'terms'], sorted by n_terms desc
    """

    df = pd.DataFrame({
        "term": clusters.index.astype(str),
        "cluster_id": clusters.astype(int).values,
    })

    table = (
        df.groupby("cluster_id", as_index=False)
          .agg(
              n_terms=("term", "size"),
              terms=("term", lambda s: ", ".join(map(str, s)))
          )
          .sort_values("n_terms", ascending=False, kind="stable")
          .reset_index(drop=True)
    )

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, sep=";", index=False)

    return table


def cluster_stats(labels):
    counts = Counter(labels.values)  # cluster_id -> size
    sizes = pd.Series(counts).sort_values(ascending=False)
    n = int(sizes.sum())
    return {
        "n_clusters": int(sizes.size),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "singleton_ratio": float((sizes == 1).sum()) / n,
        "largest_ratio": float(sizes.iloc[0]) / n,
        "top8_coverage": float(sizes.iloc[:8].sum()) / n if sizes.size >= 8 else float(sizes.sum()) / n,
    }


def plot_silhouette_curve(silhouette_df, k_star=None):
    """
    Plot mean silhouette scores as a function of k.

    Args:
        silhouette_df: DataFrame
        k_star: Optional, highlight chosen k with vertical line
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(silhouette_df["k"], silhouette_df["silhouette"], marker="o", label="mean silhouette")
    if k_star is not None:
        ax.axvline(k_star, color="red", linestyle="--", label=f"chosen k={k_star}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean Silhouette score")
    ax.set_title("Silhouette score vs k")
    ax.legend()
    plt.show()