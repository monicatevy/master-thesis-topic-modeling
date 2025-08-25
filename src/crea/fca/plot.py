from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from wordcloud import WordCloud

def plot_heatmap(matrix: pd.DataFrame,
                 title: str,
                 cbar_label: str,
                 as_percent: bool = False,
                 x_label: str | None = None,
                 y_label: str | None = None,
                 save_path: str | None = None):
    """
    Plot a heatmap from a numeric DataFrame.
    Args:
      matrix: DataFrame to plot
      as_percent: Multiply values by 100 before plotting.
      title: Figure title
      cbar_label: Label for the colorbar
      x_label: Optional X-axis label
      y_label: Optional Y-axis label
      save_path: Optional path to save the PNG
    Returns:
      None (optionally saves figure)
    """

    data = matrix * 100 if as_percent else matrix
    fmt = ".0f" if as_percent else ".2f"

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Reds",
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def compute_clusters_from_similarity(
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


def plot_dendrogram(
    Z,
    labels: list[str],
    title: str,
    threshold: float | None = None,
    x_label: str | None = "Distance",
    y_label: str | None = "Objects",
    save_path: str | None = None,
):
    """
    Render a dendrogram from a SciPy linkage matrix
    Args:
      Z: SciPy linkage matrix
      labels: Leaf labels as the original data
      title: Figure title
      threshold: Optional vertical cut line to display on the plot
      x_label: Optional X-axis label
      y_label: Optional Y-axis label
      save_path: Optional path to save the PNG
    Returns:
      None (optionally saves figure)
    """
    plt.figure(figsize=(12, 8))
    dendrogram(
        Z,
        labels=labels,
        orientation="left",
        leaf_font_size=10,
        color_threshold=None,
    )
    if threshold is not None:
        plt.axvline(x=threshold, color="red", linestyle="--", label=f"Cut at {threshold}")

    plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if threshold is not None:
        plt.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_distance_distribution(sim_matrix: pd.DataFrame, bins: int = 50):
    """
    Plot histogram and KDE of pairwise distances from a similarity matrix
    Args:
      sim_matrix: Square DataFrame of similarities in [0,1]
      bins: Number of bins in the histogram
    Returns:
      distances: Flattened array of unique pairwise distances
    """
    similarity = sim_matrix.astype(float).copy()
    dist = 1.0 - similarity

    # Ne garder que la partie supérieure triangulaire (sans la diagonale)
    distances = dist.where(np.triu(np.ones(dist.shape), k=1).astype(bool)).stack().values

    plt.figure(figsize=(10,6))
    sns.histplot(distances, bins=bins, kde=True)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Pairwise Distances")
    plt.tight_layout()
    plt.show()

    return distances


def plot_wordclouds_from_clusters(
    clusters: pd.Series,
    max_clusters: int | None = None,
    words_per_cluster_limit: int = 50,
    title_prefix: str = "Cluster",
    cluster_order: list | None = None,
    save_path: str | None = None
) -> None:
    """
    Plot one word cloud per cluster from a Series of labels.
    Args:
      clusters: pd.Series mapping each term (index) to a cluster id (values)
      max_clusters: Max number of clusters to plot (sorted by cluster id)
      words_per_cluster_limit: Keep at most N terms per cluster
      title_prefix: Prefix for subplot titles
      cluster_order: Optional list of cluster ids to render in order
      save_path: Optional path to save the PNG
    Returns:
      None (optionally saves figure)
    """

    df = pd.DataFrame({
        "term": clusters.index.astype(str),
        "cluster_id": clusters.astype(object).values,
    })

    # Base order
    base_order = df["cluster_id"].drop_duplicates().tolist()

    # Prioritize requested clusters
    if cluster_order:
        first = [cid for cid in cluster_order if cid in base_order]
        rest = [cid for cid in base_order if cid not in first]
        cluster_ids = first + rest
    else:
        cluster_ids = base_order

    if max_clusters is not None:
        cluster_ids = cluster_ids[:max_clusters]

    if not cluster_ids:
        raise ValueError("No clusters to plot.")

    fig, axes = plt.subplots(len(cluster_ids), 1, figsize=(10, 4 * len(cluster_ids)))
    axes = np.atleast_1d(axes)

    for ax, cid in zip(axes, cluster_ids):
        terms = df.loc[df["cluster_id"] == cid, "term"].tolist()
        tokens = terms[:words_per_cluster_limit]
        text = " ".join(tokens)

        if not tokens:
            ax.text(0.5, 0.5, "No terms", ha="center", va="center")
            ax.axis("off")
            continue

        wc = WordCloud(width=800, height=200, background_color="white")
        img = wc.generate(text)

        ax.imshow(img, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{title_prefix} {cid} – {len(terms)} terms")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


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