from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA


# Effect Size

def effect_size(
    df: pd.DataFrame,
    normal_mask: pd.Series,
    attack_mask: pd.Series,
    eps: float = 1e-12,
):
    """
    Create an effect_size(col) function that computes absolute Cohen's d
    between normal and attack groups for a single column.

    This returns a closure so you can do:
        effect_size = make_effect_size(df, normal_mask, attack_mask, eps)
        X_num.columns.to_series().apply(effect_size)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the feature columns.
    normal_mask : pd.Series
        Boolean mask for normal rows (aligned to df.index).
    attack_mask : pd.Series
        Boolean mask for attack rows (aligned to df.index).
    eps : float, default=1e-12
        Stability term to avoid division by ~0.

    Returns
    -------
    callable
        Function effect_size(col: str) -> float
    """

    # Ensure masks align to df index
    normal_mask = normal_mask.reindex(df.index).fillna(False)
    attack_mask = attack_mask.reindex(df.index).fillna(False)

    def effect_size(col: str) -> float:
        # Convert to numeric and drop invalid values
        n = pd.to_numeric(df.loc[normal_mask, col], errors="coerce").dropna()
        a = pd.to_numeric(df.loc[attack_mask, col], errors="coerce").dropna()

        # If either group is empty after coercion, treat as missing
        if n.empty or a.empty:
            return np.nan

        pooled_std = np.sqrt((n.var(ddof=0) + a.var(ddof=0)) / 2)

        if not np.isfinite(pooled_std) or pooled_std < eps:
            return 0.0  # essentially no variability (or degenerate)

        return float(abs(n.mean() - a.mean()) / pooled_std)

    return effect_size



# Gap Statistic
def _sample_uniform_reference(
    X: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a reference dataset uniformly within the axis-aligned bounding box
    of X (per-feature min/max).
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return rng.random((n, X.shape[1])) * (maxs - mins) + mins


def gap_statistic_kmeans(
    X: np.ndarray,
    k_range: range,
    n_refs: int = 10,
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """
    Compute Gap Statistic for KMeans across k_range.

    Gap(k) = E_ref[log(Wk_ref)] - log(Wk)
    where Wk is KMeans inertia (within-cluster dispersion).
    Also returns sk (standard error term) from Tibshirani et al.

    Notes:
    - Works best on scaled data.
    - Reference distribution uses uniform sampling in the bounding box of X.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km.fit(X)
        Wk = km.inertia_
        logWk = np.log(Wk)

        ref_logWk = []
        for b in range(n_refs):
            X_ref = _sample_uniform_reference(X, n=n, rng=rng)
            km_ref = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            km_ref.fit(X_ref)
            ref_logWk.append(np.log(km_ref.inertia_))

        ref_logWk = np.array(ref_logWk)
        gap = ref_logWk.mean() - logWk
        sk = ref_logWk.std(ddof=1) * np.sqrt(1 + 1 / n_refs)

        rows.append({
            "k": k,
            "gap": gap,
            "sk": sk,
            "logWk": logWk,
            "ref_logWk_mean": ref_logWk.mean(),
        })

    return pd.DataFrame(rows)


def choose_k_gap_rule(gap_df: pd.DataFrame) -> Optional[int]:
    """
    Tibshirani selection rule:
    choose smallest k such that Gap(k) >= Gap(k+1) - s(k+1)

    Returns None if cannot compute (e.g., missing k+1).
    """
    df = gap_df.sort_values("k").reset_index(drop=True)
    for i in range(len(df) - 1):
        k = int(df.loc[i, "k"])
        gap_k = df.loc[i, "gap"]
        gap_k1 = df.loc[i + 1, "gap"]
        s_k1 = df.loc[i + 1, "sk"]
        if gap_k >= (gap_k1 - s_k1):
            return k
    return None


# K selection diagnostics (Elbow + Silhouette + CH + Gap)
def evaluate_kmeans_k(
    X: np.ndarray,
    k_range: range,
    random_state: int = 42,
    n_init: int = 10,
    n_refs_gap: int = 10,
) -> pd.DataFrame:
    """
    Returns a single dataframe with:
      - inertia (elbow)
      - silhouette
      - calinski_harabasz
      - gap + sk (gap statistic)
    """
    # KMeans metrics
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)

        inertia = km.inertia_
        sil = np.nan if k < 2 else silhouette_score(X, labels)
        ch = np.nan if k < 2 else calinski_harabasz_score(X, labels)

        rows.append({
            "k": k,
            "inertia": inertia,
            "silhouette": sil,
            "calinski_harabasz": ch,
        })

    metrics_df = pd.DataFrame(rows)

    # Gap statistic
    gap_df = gap_statistic_kmeans(
        X, k_range=k_range, n_refs=n_refs_gap,
        random_state=random_state, n_init=n_init
    )

    out = metrics_df.merge(gap_df[["k", "gap", "sk"]], on="k", how="left")
    return out


def plot_k_diagnostics(metrics_df: pd.DataFrame) -> None:
    """
    2x2: inertia, silhouette, CH, gap (with error bars).
    """
    df = metrics_df.sort_values("k")
    plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
    })
    fig, axes = plt.subplots(2, 2, figsize=(6, 4))

    axes[0, 0].plot(df["k"], df["inertia"], marker="o")
    axes[0, 0].set_title("KMeans Inertia (Elbow)")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Inertia")

    axes[0, 1].plot(df["k"], df["silhouette"], marker="o")
    axes[0, 1].set_title("Silhouette Score")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Silhouette Score")

    axes[1, 0].plot(df["k"], df["calinski_harabasz"], marker="o")
    axes[1, 0].set_title("Calinski–Harabasz")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("CH score")

    axes[1, 1].errorbar(df["k"], df["gap"], yerr=df["sk"], fmt="o-")
    axes[1, 1].set_title("Gap Statistic")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Gap Statistic")

    plt.tight_layout()
    plt.show()



# Fit + visualize
def fit_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    return km, labels


def pca_project(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z, pca.explained_variance_ratio_


def plot_clusters_pca(
    Z2: np.ndarray,
    labels: np.ndarray,
    evr: np.ndarray,
    title: str = "Clusters in PCA space",
    s: int = 6,
    alpha: float = 0.5,
) -> None:
    plt.figure(figsize=(4, 4))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=s, alpha=alpha)
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    plt.title(title)
    plt.tight_layout()
    plt.show()