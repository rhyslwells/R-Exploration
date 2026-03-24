"""
================================================================================
DATASET COMPARISON FRAMEWORK
================================================================================

OVERVIEW:
---------
This script demonstrates three fundamental approaches to comparing multiple
dataset partitions drawn from the same global distribution:

    1. Summary Statistics Comparison
       - Compute descriptive statistic vectors φ(D_i) per dataset
       - Compare via Euclidean distance matrix and PCA

    2. Distributional Comparison
       - Kolmogorov-Smirnov test (per feature, 1D)
       - Wasserstein distance (per feature, 1D)
       - Maximum Mean Discrepancy (multivariate)

    3. Model-Based Comparison
       - Fit linear regression f_i(x) = β_i^T x to each partition
       - Compare: (a) parameters β_i, (b) cross-evaluation performance

DATASET:
--------
We use the sklearn `wine` dataset (178 samples, 13 features).
We select 4 features and partition into D_0, D_1, ..., D_{k-1} equal-size
random samples. All partitions share the same variables and scales.

    Features selected:
        - alcohol
        - malic_acid
        - ash
        - magnesium

    Target: wine class (0, 1, 2) — used as regression target to give
            the model-fitting section a meaningful signal.

STRUCTURE:
----------
    Section 0 : Data loading and partitioning
    Section 1 : Summary statistics comparison
    Section 2 : Distributional comparison
    Section 3 : Model-based comparison
    Section 4 : Visualisation dashboard

================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Numerical / statistical ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance

# ── Machine learning ──────────────────────────────────────────────────────────
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10")
PALETTE = sns.color_palette("tab10")


# ==============================================================================
# SECTION 0 — DATA LOADING AND PARTITIONING
# ==============================================================================
"""
We load the Wine dataset and select 4 features.
The full dataset is shuffled and split into K equal-size partitions.

Each partition D_i is a tuple (X_i, y_i) where:
    X_i : (n_i, 4) feature matrix
    y_i : (n_i,)  target vector (wine class)

Partition size n_i = floor(N / K), so the last few samples may be discarded
to keep all partitions exactly equal in size — ensuring fair comparison.
"""

def load_and_partition(n_partitions: int = 4) -> tuple:
    """
    Load the Wine dataset, select 4 features, and split into equal partitions.

    Parameters
    ----------
    n_partitions : int
        Number of equal-size dataset partitions K.

    Returns
    -------
    partitions : list of dict
        Each element is {"label": str, "X": ndarray, "y": ndarray}.
    feature_names : list of str
        Names of the 4 selected features.
    """
    wine = load_wine()
    feature_names = ["alcohol", "malic_acid", "ash", "magnesium"]
    feature_indices = [wine.feature_names.index(f) for f in feature_names]

    X_full = wine.data[:, feature_indices]
    y_full = wine.target.astype(float)

    # Shuffle
    idx = np.random.permutation(len(y_full))
    X_full, y_full = X_full[idx], y_full[idx]

    # Trim to exact multiple of K
    n_per = len(y_full) // n_partitions
    N_used = n_per * n_partitions
    X_full, y_full = X_full[:N_used], y_full[:N_used]

    partitions = []
    for k in range(n_partitions):
        sl = slice(k * n_per, (k + 1) * n_per)
        partitions.append({
            "label": f"D_{k}",
            "X": X_full[sl],
            "y": y_full[sl],
        })

    print(f"[Section 0] Loaded Wine dataset.")
    print(f"            Features : {feature_names}")
    print(f"            Partitions: {n_partitions}  |  Samples per partition: {n_per}")
    print(f"            Total samples used: {N_used} / {len(wine.target)}\n")

    return partitions, feature_names


# ==============================================================================
# SECTION 1 — SUMMARY STATISTICS COMPARISON
# ==============================================================================
"""
THEORY:
-------
For each dataset D_i we compute a statistic vector:

    φ(D_i) = [μ_1, σ²_1, skew_1, kurt_1,  ← feature 1
               μ_2, σ²_2, skew_2, kurt_2,  ← feature 2
               ...                          ← features 3, 4
               n_i]                         ← sample size

Comparison methods:
    (a) Pairwise Euclidean distance matrix  ||φ(D_i) - φ(D_j)||
    (b) PCA on the matrix [φ(D_0), φ(D_1), ...] to visualise spread

Interpretation:
    - Small distance → datasets are statistically similar
    - Large distance → datasets differ in location, spread, or shape
    - PCA shows which statistic dimensions drive the most variation
"""

def compute_stat_vector(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """
    Compute the summary statistic vector φ(D_i) for a single partition.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    feature_names : list of str

    Returns
    -------
    stats_dict : dict
        Flat dictionary of all statistics (used for display and distance).
    """
    records = {}
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        records[f"{fname}_mean"]     = np.mean(col)
        records[f"{fname}_median"]   = np.median(col)
        records[f"{fname}_var"]      = np.var(col, ddof=1)
        records[f"{fname}_iqr"]      = stats.iqr(col)
        records[f"{fname}_skewness"] = stats.skew(col)
        records[f"{fname}_kurtosis"] = stats.kurtosis(col)
    records["n"] = len(y)
    return records


def summary_statistics_comparison(partitions: list, feature_names: list) -> pd.DataFrame:
    """
    Compute φ(D_i) for all partitions, print a comparison table,
    and return the distance matrix.

    Parameters
    ----------
    partitions : list of dict
    feature_names : list of str

    Returns
    -------
    stat_df : pd.DataFrame
        Rows = partitions, columns = statistics.
    dist_matrix : pd.DataFrame
        Pairwise Euclidean distance between φ vectors.
    """
    print("=" * 70)
    print("SECTION 1 — SUMMARY STATISTICS COMPARISON")
    print("=" * 70)

    stat_rows = {}
    for p in partitions:
        stat_rows[p["label"]] = compute_stat_vector(p["X"], p["y"], feature_names)

    stat_df = pd.DataFrame(stat_rows).T  # rows=datasets, cols=stats

    print("\nStatistic vectors φ(D_i)  [rows=datasets, cols=statistics]:")
    print(stat_df.round(3).to_string())

    # ── Pairwise Euclidean distance ───────────────────────────────────────────
    # Normalise before computing distance so that high-variance stats
    # (e.g. magnesium variance) don't dominate.
    scaler = StandardScaler()
    phi_scaled = scaler.fit_transform(stat_df.values)

    labels = stat_df.index.tolist()
    K = len(labels)
    dist_matrix = pd.DataFrame(np.zeros((K, K)), index=labels, columns=labels)

    for i in range(K):
        for j in range(K):
            dist_matrix.iloc[i, j] = np.linalg.norm(phi_scaled[i] - phi_scaled[j])

    print("\nPairwise Euclidean distance  ||φ(D_i) - φ(D_j)||  (normalised):")
    print(dist_matrix.round(4).to_string())
    print()

    return stat_df, dist_matrix


# ==============================================================================
# SECTION 2 — DISTRIBUTIONAL COMPARISON
# ==============================================================================
"""
THEORY:
-------
Rather than summarising each dataset into a single vector, we compare the
empirical distributions directly.

(A) Kolmogorov–Smirnov test  (per feature, 1D)
    ─────────────────────────────────────────────
    H₀: D_i and D_j are drawn from the same distribution for feature k.
    Statistic: KS = sup_x |F_i(x) - F_j(x)|
    p-value < 0.05 → reject H₀ (distributions differ significantly).

(B) Wasserstein distance  (per feature, 1D)
    ─────────────────────────────────────────
    W(P_i, P_j) = ∫ |F_i(x) - F_j(x)| dx
    Measures the "earth-mover" cost to transform one distribution into another.
    Unlike KS, this is a proper metric and quantifies *how much* they differ.

(C) Maximum Mean Discrepancy  (multivariate)
    ─────────────────────────────────────────
    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    where k is an RBF kernel.  MMD = 0 iff P = Q.
    This captures multivariate structure that per-feature tests miss.
"""

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """
    Compute the Maximum Mean Discrepancy with an RBF kernel.

    MMD²(X, Y) = mean(k(X,X)) - 2·mean(k(X,Y)) + mean(k(Y,Y))
    where k(a,b) = exp(-γ ||a-b||²)

    Parameters
    ----------
    X, Y : ndarray, shape (n, p)
    gamma : float or None
        RBF bandwidth.  If None, uses the median heuristic: γ = 1/(2·σ²)
        where σ² is the median pairwise squared distance.

    Returns
    -------
    mmd : float  (≥ 0)
    """
    if gamma is None:
        # Median heuristic on the pooled sample
        Z = np.vstack([X, Y])
        sq_dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
        median_sq = np.median(sq_dists[sq_dists > 0])
        gamma = 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0

    def rbf_kernel_mean(A, B):
        sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.mean(np.exp(-gamma * sq))

    return rbf_kernel_mean(X, X) - 2 * rbf_kernel_mean(X, Y) + rbf_kernel_mean(Y, Y)


def distributional_comparison(partitions: list, feature_names: list) -> dict:
    """
    Perform pairwise distributional comparison across all partitions.

    Returns
    -------
    results : dict with keys "ks", "wasserstein", "mmd"
        Each value is a pd.DataFrame (pairwise matrix).
    """
    print("=" * 70)
    print("SECTION 2 — DISTRIBUTIONAL COMPARISON")
    print("=" * 70)

    labels = [p["label"] for p in partitions]
    K = len(partitions)
    p_feat = len(feature_names)

    # Storage: one matrix per feature for KS and Wasserstein
    ks_stat   = {f: pd.DataFrame(np.zeros((K, K)), index=labels, columns=labels)
                 for f in feature_names}
    ks_pval   = {f: pd.DataFrame(np.ones((K, K)),  index=labels, columns=labels)
                 for f in feature_names}
    wass_dist = {f: pd.DataFrame(np.zeros((K, K)), index=labels, columns=labels)
                 for f in feature_names}
    mmd_mat   = pd.DataFrame(np.zeros((K, K)), index=labels, columns=labels)

    for i in range(K):
        for j in range(i + 1, K):
            Xi, Xj = partitions[i]["X"], partitions[j]["X"]

            # (A) KS test and (B) Wasserstein — per feature
            for fi, fname in enumerate(feature_names):
                col_i, col_j = Xi[:, fi], Xj[:, fi]

                ks_res = ks_2samp(col_i, col_j)
                ks_stat[fname].iloc[i, j] = ks_stat[fname].iloc[j, i] = ks_res.statistic
                ks_pval[fname].iloc[i, j] = ks_pval[fname].iloc[j, i] = ks_res.pvalue

                w = wasserstein_distance(col_i, col_j)
                wass_dist[fname].iloc[i, j] = wass_dist[fname].iloc[j, i] = w

            # (C) MMD — multivariate
            mmd_val = mmd_rbf(Xi, Xj)
            mmd_mat.iloc[i, j] = mmd_mat.iloc[j, i] = mmd_val

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n(A) Kolmogorov–Smirnov statistic  (per feature)")
    print("    Interpretation: higher → CDFs differ more; p<0.05 → significant\n")
    for fname in feature_names:
        print(f"  Feature: {fname}")
        print(f"    KS statistic:\n{ks_stat[fname].round(4).to_string()}")
        print(f"    p-values:\n{ks_pval[fname].round(4).to_string()}\n")

    print("\n(B) Wasserstein distance  (per feature)")
    print("    Interpretation: earth-mover cost; 0 = identical distributions\n")
    for fname in feature_names:
        print(f"  Feature: {fname}")
        print(wass_dist[fname].round(4).to_string(), "\n")

    print("\n(C) Maximum Mean Discrepancy  (multivariate, RBF kernel)")
    print("    Interpretation: 0 = identical joint distributions\n")
    print(mmd_mat.round(6).to_string(), "\n")

    return {"ks_stat": ks_stat, "ks_pval": ks_pval,
            "wasserstein": wass_dist, "mmd": mmd_mat}


# ==============================================================================
# SECTION 3 — MODEL-BASED COMPARISON
# ==============================================================================
"""
THEORY:
-------
Fit a linear regression model f_i(x) = β_i^T x to each partition D_i.

(A) Parameter comparison
    ─────────────────────
    Compare coefficient vectors β_i directly.
    If β_i ≈ β_j → datasets support the same linear relationship.
    Distance: ||β_i - β_j||₂

(B) Functional similarity
    ──────────────────────
    Evaluate both models on a shared grid X_ref and compute:
        d_func(f_i, f_j) = mean( (f_i(X_ref) - f_j(X_ref))² )
    This captures how differently the models *behave* across the input space,
    even if individual coefficients look similar.

(C) Cross-evaluation
    ─────────────────
    Train on D_i, evaluate on D_j.
    If MSE(train=i, test=j) ≈ MSE(train=i, test=i)
        → D_j is consistent with D_i's generating process.
    If MSE degrades sharply → structural difference between partitions.
"""

def model_based_comparison(partitions: list, feature_names: list) -> dict:
    """
    Fit linear regression to each partition and compare via:
        (A) parameter distance, (B) functional distance, (C) cross-evaluation.

    Returns
    -------
    results : dict with keys "coef_df", "param_dist", "func_dist", "cross_mse"
    """
    print("=" * 70)
    print("SECTION 3 — MODEL-BASED COMPARISON")
    print("=" * 70)

    labels = [p["label"] for p in partitions]
    K = len(partitions)

    # ── Fit one model per partition ───────────────────────────────────────────
    models = []
    coef_rows = {}

    for p in partitions:
        model = LinearRegression()
        model.fit(p["X"], p["y"])
        models.append(model)
        coef_rows[p["label"]] = dict(zip(feature_names, model.coef_))
        coef_rows[p["label"]]["intercept"] = model.intercept_

    coef_df = pd.DataFrame(coef_rows).T
    print("\n(A) Fitted coefficients  β_i  (rows=datasets, cols=features):")
    print(coef_df.round(4).to_string(), "\n")

    # ── (A) Parameter distance  ||β_i - β_j||₂ ───────────────────────────────
    param_dist = pd.DataFrame(np.zeros((K, K)), index=labels, columns=labels)
    for i in range(K):
        for j in range(K):
            diff = models[i].coef_ - models[j].coef_
            param_dist.iloc[i, j] = np.linalg.norm(diff)

    print("    Pairwise parameter distance  ||β_i - β_j||₂:")
    print(param_dist.round(4).to_string(), "\n")

    # ── (B) Functional distance on shared reference grid ─────────────────────
    # X_ref = mean of all partitions stacked (representative input space)
    X_all = np.vstack([p["X"] for p in partitions])
    X_ref = X_all  # use all observed points as the shared domain

    func_dist = pd.DataFrame(np.zeros((K, K)), index=labels, columns=labels)
    preds = [m.predict(X_ref) for m in models]

    for i in range(K):
        for j in range(K):
            func_dist.iloc[i, j] = np.mean((preds[i] - preds[j]) ** 2)

    print("    Functional distance  mean( (f_i(X_ref) - f_j(X_ref))² ):")
    print(func_dist.round(6).to_string(), "\n")

    # ── (C) Cross-evaluation MSE ──────────────────────────────────────────────
    # cross_mse[i, j] = MSE when model trained on D_i is tested on D_j
    cross_mse = pd.DataFrame(np.zeros((K, K)), index=labels, columns=labels)

    for i in range(K):
        for j in range(K):
            y_pred = models[i].predict(partitions[j]["X"])
            cross_mse.iloc[i, j] = mean_squared_error(partitions[j]["y"], y_pred)

    print("    Cross-evaluation MSE  (row=train partition, col=test partition):")
    print("    Diagonal = in-sample MSE; off-diagonal = out-of-sample MSE")
    print(cross_mse.round(4).to_string(), "\n")

    # Highlight degradation: ratio of off-diagonal to diagonal
    diag_mse = np.diag(cross_mse.values)
    degradation = cross_mse.copy()
    for i in range(K):
        degradation.iloc[i, :] = cross_mse.iloc[i, :] / diag_mse[i]

    print("    MSE degradation ratio  (off-diag / diag);  1.0 = no degradation:")
    print(degradation.round(3).to_string(), "\n")

    return {
        "coef_df":    coef_df,
        "param_dist": param_dist,
        "func_dist":  func_dist,
        "cross_mse":  cross_mse,
        "degradation": degradation,
    }


# ==============================================================================
# SECTION 4 — VISUALISATION DASHBOARD
# ==============================================================================

def plot_dashboard(partitions, feature_names, stat_df, dist_matrix,
                   dist_results, model_results):
    """
    Produce a multi-panel visualisation dashboard covering all three
    comparison approaches.

    Layout
    ------
    Row 0 : Feature distributions (KDE per partition, one panel per feature)
    Row 1 : [Stat distance heatmap] [PCA of φ vectors] [MMD heatmap]
    Row 2 : [Coefficient bar chart] [Cross-eval MSE heatmap] [Wasserstein heatmap]
    """
    K = len(partitions)
    p = len(feature_names)
    labels = [part["label"] for part in partitions]
    colors = [PALETTE[i] for i in range(K)]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Dataset Comparison Framework — Wine Dataset Partitions",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    # ── Row 0: KDE plots per feature ─────────────────────────────────────────
    for fi, fname in enumerate(feature_names):
        ax = fig.add_subplot(gs[0, fi])
        for ki, part in enumerate(partitions):
            sns.kdeplot(part["X"][:, fi], ax=ax, label=part["label"],
                        color=colors[ki], linewidth=2)
        ax.set_title(f"Distribution: {fname}", fontsize=10)
        ax.set_xlabel(fname, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        if fi == 0:
            ax.legend(fontsize=7)

    # ── Row 1, col 0: Summary stat distance heatmap ──────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    sns.heatmap(dist_matrix.astype(float), annot=True, fmt=".2f",
                cmap="YlOrRd", ax=ax1, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax1.set_title("§1  ||φ(D_i) − φ(D_j)||₂\n(normalised stat distance)", fontsize=9)

    # ── Row 1, col 1: PCA of φ vectors ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    scaler = StandardScaler()
    phi_scaled = scaler.fit_transform(stat_df.values)
    # PCA is only meaningful with ≥2 components; with K=4 datasets we have 4 points
    n_components = min(2, phi_scaled.shape[0], phi_scaled.shape[1])
    pca = PCA(n_components=n_components)
    phi_pca = pca.fit_transform(phi_scaled)
    for ki in range(K):
        ax2.scatter(phi_pca[ki, 0], phi_pca[ki, 1] if n_components > 1 else 0,
                    color=colors[ki], s=120, zorder=3, label=labels[ki])
        ax2.annotate(labels[ki], (phi_pca[ki, 0],
                                   phi_pca[ki, 1] if n_components > 1 else 0),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
                   if n_components > 1 else "—", fontsize=8)
    ax2.set_title("§1  PCA of φ vectors\n(variation across datasets)", fontsize=9)
    ax2.legend(fontsize=7)

    # ── Row 1, col 2: MMD heatmap ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    sns.heatmap(dist_results["mmd"].astype(float), annot=True, fmt=".5f",
                cmap="Blues", ax=ax3, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax3.set_title("§2  MMD (multivariate)\nRBF kernel", fontsize=9)

    # ── Row 1, col 3: KS statistic heatmap (first feature) ───────────────────
    ax4 = fig.add_subplot(gs[1, 3])
    ks_first = dist_results["ks_stat"][feature_names[0]].astype(float)
    sns.heatmap(ks_first, annot=True, fmt=".3f",
                cmap="Greens", ax=ax4, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax4.set_title(f"§2  KS statistic\nfeature: {feature_names[0]}", fontsize=9)

    # ── Row 2, col 0-1: Coefficient bar chart ────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    coef_df = model_results["coef_df"][feature_names]  # exclude intercept
    x = np.arange(len(feature_names))
    width = 0.8 / K
    for ki, label in enumerate(labels):
        offset = (ki - K / 2 + 0.5) * width
        ax5.bar(x + offset, coef_df.loc[label].values,
                width=width, label=label, color=colors[ki], alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(feature_names, fontsize=9)
    ax5.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax5.set_title("§3  Linear regression coefficients β_i per partition", fontsize=9)
    ax5.set_ylabel("Coefficient value", fontsize=8)
    ax5.legend(fontsize=8)

    # ── Row 2, col 2: Cross-evaluation MSE heatmap ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    sns.heatmap(model_results["cross_mse"].astype(float), annot=True, fmt=".3f",
                cmap="OrRd", ax=ax6, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax6.set_title("§3  Cross-eval MSE\n(row=train, col=test)", fontsize=9)
    ax6.set_xlabel("Test partition", fontsize=8)
    ax6.set_ylabel("Train partition", fontsize=8)

    # ── Row 2, col 3: Wasserstein heatmap (first feature) ────────────────────
    ax7 = fig.add_subplot(gs[2, 3])
    wass_first = dist_results["wasserstein"][feature_names[0]].astype(float)
    sns.heatmap(wass_first, annot=True, fmt=".3f",
                cmap="Purples", ax=ax7, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax7.set_title(f"§2  Wasserstein distance\nfeature: {feature_names[0]}", fontsize=9)

    plt.savefig("dataset_comparison_dashboard.png", dpi=150, bbox_inches="tight")
    print("[Section 4] Dashboard saved → dataset_comparison_dashboard.png")
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # ── 0. Load and partition ─────────────────────────────────────────────────
    partitions, feature_names = load_and_partition(n_partitions=4)

    # ── 1. Summary statistics ─────────────────────────────────────────────────
    stat_df, dist_matrix = summary_statistics_comparison(partitions, feature_names)

    # ── 2. Distributional comparison ──────────────────────────────────────────
    dist_results = distributional_comparison(partitions, feature_names)

    # ── 3. Model-based comparison ─────────────────────────────────────────────
    model_results = model_based_comparison(partitions, feature_names)

    # ── 4. Visualisation ──────────────────────────────────────────────────────
    plot_dashboard(partitions, feature_names, stat_df, dist_matrix,
                   dist_results, model_results)

    # ── Summary interpretation ────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — WHAT TO LOOK FOR")
    print("=" * 70)
    print("""
  §1  Stat distance / PCA
      • All distances small, points clustered in PCA
        → partitions are statistically homogeneous (expected for random splits)

  §2  KS / Wasserstein / MMD
      • KS p-values >> 0.05 → cannot reject same-distribution hypothesis
      • Wasserstein ≈ 0     → distributions nearly identical
      • MMD ≈ 0             → joint distributions match

  §3  Model comparison
      • β_i vectors similar  → same linear structure across partitions
      • Cross-eval MSE ≈ diagonal MSE → models generalise across partitions
      • Degradation ratio ≈ 1.0       → no structural difference detected

  If you introduce a *biased* partition (e.g. only class-0 wines), all three
  sections will flag it: large stat distance, high KS/Wasserstein/MMD, and
  sharp cross-eval MSE degradation.
    """)


if __name__ == "__main__":
    main()