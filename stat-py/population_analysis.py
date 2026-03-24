"""
================================================================================
POPULATION OF DATASETS — STRUCTURAL ANALYSIS
================================================================================

CONTEXT:
--------
In the previous script (Section 1–3) we compared datasets PAIRWISE:
    D_i vs D_j → a single distance or test statistic

Now we ask a different question:
    "What is the STRUCTURE of the collection {D_0, D_1, D_2, D_3}?"

This is a shift in thinking:

    PAIRWISE:     "How different are D_i and D_j?"
                   → produces a distance matrix

    POPULATION:   "What is the distribution of properties across {D_i}?"
                   → produces a model of variation

We are no longer comparing objects.
We are modelling a DISTRIBUTION OVER DATASETS.

SECTIONS:
---------
    02  Hierarchical model         — μ_0, Σ_0  (average dataset + spread)
    03  Global model + deviation   — θ_global + δ_i
    04  Pooling comparison         — no / full / partial pooling
    05  Distribution over coefs    — β^(i) as random variables
    06  Functional alignment       — PCA on prediction vectors F_i

DATASET:
--------
    Same Wine dataset partitions from the pairwise script.
    4 features: alcohol, malic_acid, ash, magnesium
    4 equal-size random partitions: D_0, D_1, D_2, D_3
    Target: wine class (0, 1, 2)
================================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="tab10")
PALETTE = sns.color_palette("tab10")

# ── Reuse partition loader from pairwise script ───────────────────────────────
def load_and_partition(n_partitions=4):
    wine         = load_wine()
    feature_names = ["alcohol", "malic_acid", "ash", "magnesium"]
    feat_idx     = [wine.feature_names.index(f) for f in feature_names]
    X_full       = wine.data[:, feat_idx]
    y_full       = wine.target.astype(float)
    idx          = np.random.permutation(len(y_full))
    X_full, y_full = X_full[idx], y_full[idx]
    n_per        = len(y_full) // n_partitions
    X_full       = X_full[:n_per * n_partitions]
    y_full       = y_full[:n_per * n_partitions]
    partitions   = [{"label": f"D_{k}",
                     "X": X_full[k*n_per:(k+1)*n_per],
                     "y": y_full[k*n_per:(k+1)*n_per]}
                    for k in range(n_partitions)]
    return partitions, feature_names

partitions, feature_names = load_and_partition(n_partitions=4)
labels = [p["label"] for p in partitions]
K      = len(partitions)


# ==============================================================================
# SECTION 02 — HIERARCHICAL MODEL
# ==============================================================================
"""
THEORY:
-------
Instead of treating each dataset independently, assume they are drawn
from a common population:

    D_i ~ P(θ_i)       each dataset has its own parameters
    θ_i ~ P(θ)         those parameters come from a shared distribution

In the Gaussian case:

    X | D_i ~ N(μ_i, Σ_i)      dataset-level distribution
    μ_i     ~ N(μ_0, Σ_0)      population-level distribution

We estimate two levels:

    DATASET LEVEL:    μ_i, Σ_i  for each D_i
    POPULATION LEVEL: μ_0 = mean of {μ_i}
                      Σ_0 = covariance of {μ_i}

INSIGHT:
--------
    μ_0       = the "average dataset" — what a typical D_i looks like
    Σ_0       = how datasets vary from each other
    Σ_0[j,j]  = variance of feature j's mean across datasets
                 large → feature j is unstable across datasets
                 small → feature j is consistent across datasets
    Σ_0[j,k]  = when one dataset has high mean on feature j,
                 does it also tend to have high mean on feature k?

This is the most principled way to answer:
    "What is the typical dataset and how much do they vary?"
"""

def section_02_hierarchical(partitions, feature_names):
    print("=" * 70)
    print("SECTION 02 — HIERARCHICAL MODEL")
    print("μ_0 = average dataset | Σ_0 = how datasets vary")
    print("=" * 70)

    # ── Step 1: Compute dataset-level parameters ──────────────────────────
    # For each D_i compute μ_i (mean vector) and Σ_i (covariance matrix)
    mu_list    = []   # list of mean vectors,     shape (K, p)
    sigma_list = []   # list of covariance matrices, shape (K, p, p)

    for p in partitions:
        mu_i    = np.mean(p["X"], axis=0)          # shape (p,)
        sigma_i = np.cov(p["X"].T)                 # shape (p, p)
        mu_list.append(mu_i)
        sigma_list.append(sigma_i)
        print(f"  {p['label']}  μ_i = {np.round(mu_i, 3)}")

    mu_array = np.array(mu_list)   # shape (K, p)

    # ── Step 2: Compute population-level parameters ───────────────────────
    # μ_0 = mean of the dataset means — the "average dataset"
    # Σ_0 = covariance of the dataset means — how datasets differ
    mu_0    = np.mean(mu_array, axis=0)     # shape (p,)
    Sigma_0 = np.cov(mu_array.T)            # shape (p, p)

    print(f"\n  Population mean  μ_0  = {np.round(mu_0, 3)}")
    print(f"\n  Population covariance Σ_0 (how dataset means vary):")
    print(pd.DataFrame(Sigma_0,
                       index=feature_names,
                       columns=feature_names).round(4).to_string())

    # ── Step 3: Visualise ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Section 02 — Hierarchical Model\n"
                 "Dataset-level means vs population mean  |  "
                 "Population covariance Σ_0  |  "
                 "Spread of μ_i around μ_0",
                 fontsize=11, fontweight="bold")

    # Panel 0: μ_i per feature with μ_0 overlaid
    # Each group of bars = one feature, each bar = one dataset
    # The horizontal line = μ_0 (population mean)
    ax = axes[0]
    x  = np.arange(len(feature_names))
    w  = 0.8 / K
    for ki, p in enumerate(partitions):
        ax.bar(x + ki * w, mu_list[ki], width=w,
               label=p["label"], color=PALETTE[ki], alpha=0.8)
    for fi, fname in enumerate(feature_names):
        ax.hlines(mu_0[fi], fi - 0.1, fi + 0.8,
                  colors="black", linewidths=2,
                  linestyles="--", label="μ_0" if fi == 0 else "")
    ax.set_xticks(x + w * (K - 1) / 2)
    ax.set_xticklabels(feature_names, rotation=15, fontsize=8)
    ax.set_title("μ_i per dataset\n(dashed = population mean μ_0)", fontsize=9)
    ax.set_ylabel("Feature mean", fontsize=8)
    ax.legend(fontsize=7)

    # Panel 1: Σ_0 heatmap
    # Diagonal = variance of each feature's mean across datasets
    # Off-diagonal = covariance between feature means across datasets
    ax2 = axes[1]
    sns.heatmap(Sigma_0,
                ax=ax2,
                annot=True, fmt=".4f",
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap="coolwarm", center=0,
                linewidths=0.5,
                cbar_kws={"label": "Covariance"})
    ax2.set_title("Population covariance Σ_0\n"
                  "Diagonal = variance of feature mean across datasets\n"
                  "Large value → feature is unstable across datasets",
                  fontsize=9)

    # Panel 2: Deviation of each μ_i from μ_0
    # Shows which datasets are above/below the population average
    # per feature
    ax3 = axes[2]
    deviations = mu_array - mu_0   # shape (K, p)
    dev_df     = pd.DataFrame(deviations,
                               index=labels,
                               columns=feature_names)
    dev_df.T.plot(kind="bar", ax=ax3,
                  color=[PALETTE[i] for i in range(K)],
                  alpha=0.85, edgecolor="white")
    ax3.axhline(0, color="black", linewidth=1, linestyle="--")
    ax3.set_title("μ_i − μ_0  (deviation from population mean)\n"
                  "Positive = dataset mean above average\n"
                  "Negative = dataset mean below average",
                  fontsize=9)
    ax3.set_ylabel("Deviation", fontsize=8)
    ax3.set_xticklabels(feature_names, rotation=15, fontsize=8)
    ax3.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("02_hierarchical.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {"mu_list": mu_list, "sigma_list": sigma_list,
            "mu_0": mu_0, "Sigma_0": Sigma_0}


# ==============================================================================
# SECTION 03 — GLOBAL MODEL + DATASET-SPECIFIC DEVIATION
# ==============================================================================
"""
THEORY:
-------
Assume there is ONE underlying relationship, but each dataset shifts it:

    y = f(x; θ_i)

    θ_i = θ_global + δ_i

Where:
    θ_global = the shared structure (fit on pooled data)
    δ_i      = dataset-specific deviation from that shared structure

This is the regression version of the hierarchical model.
It is the foundation of:
    - Mixed-effects models
    - Multi-task learning
    - Hierarchical Bayesian regression

PROCEDURE:
----------
    1. Fit f_global on D = ∪ D_i  → β_global
    2. Fit f_i on each D_i         → β_i
    3. Compute δ_i = β_i - β_global

INSIGHT:
--------
    Small δ_i → D_i is well-explained by the global model
    Large δ_i → D_i has genuinely different structure
    Consistent sign of δ_i across datasets → systematic bias
    δ_i ≈ 0 for all i → full pooling is justified
    δ_i large and variable → partial pooling or no pooling needed
"""

def section_03_global_deviation(partitions, feature_names):
    print("=" * 70)
    print("SECTION 03 — GLOBAL MODEL + DATASET-SPECIFIC DEVIATION")
    print("θ_i = θ_global + δ_i")
    print("=" * 70)

    # ── Step 1: Fit global model on pooled data ───────────────────────────
    X_pool = np.vstack([p["X"] for p in partitions])
    y_pool = np.concatenate([p["y"] for p in partitions])

    global_model = LinearRegression()
    global_model.fit(X_pool, y_pool)
    beta_global  = global_model.coef_   # shape (p,)

    print(f"\n  β_global (fit on all data): {np.round(beta_global, 4)}")

    # ── Step 2: Fit per-dataset models ────────────────────────────────────
    beta_list  = []
    delta_list = []

    for p in partitions:
        model_i = LinearRegression()
        model_i.fit(p["X"], p["y"])
        beta_i  = model_i.coef_
        delta_i = beta_i - beta_global
        beta_list.append(beta_i)
        delta_list.append(delta_i)
        print(f"  {p['label']}  β_i={np.round(beta_i,4)}  "
              f"δ_i={np.round(delta_i,4)}")

    beta_array  = np.array(beta_list)    # shape (K, p)
    delta_array = np.array(delta_list)   # shape (K, p)

    # ── Step 3: Visualise ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Section 03 — Global Model + Dataset Deviation\n"
                 "θ_i = θ_global + δ_i",
                 fontsize=11, fontweight="bold")

    x = np.arange(len(feature_names))
    w = 0.8 / K

    # Panel 0: β_i per dataset with β_global overlaid
    # Shows how each dataset's coefficients compare to the global fit
    ax = axes[0]
    for ki, p in enumerate(partitions):
        ax.bar(x + ki * w, beta_list[ki], width=w,
               label=p["label"], color=PALETTE[ki], alpha=0.8)
    for fi in range(len(feature_names)):
        ax.hlines(beta_global[fi], fi - 0.05, fi + 0.85,
                  colors="black", linewidths=2.5,
                  linestyles="--", label="β_global" if fi == 0 else "")
    ax.set_xticks(x + w * (K - 1) / 2)
    ax.set_xticklabels(feature_names, rotation=15, fontsize=8)
    ax.set_title("β_i per dataset\n(dashed = β_global)", fontsize=9)
    ax.set_ylabel("Coefficient value", fontsize=8)
    ax.legend(fontsize=7)

    # Panel 1: δ_i = β_i - β_global
    # Positive bar → dataset pulls this coefficient UP from global
    # Negative bar → dataset pulls this coefficient DOWN from global
    ax2 = axes[1]
    for ki, p in enumerate(partitions):
        ax2.bar(x + ki * w, delta_list[ki], width=w,
                label=p["label"], color=PALETTE[ki], alpha=0.8)
    ax2.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax2.set_xticks(x + w * (K - 1) / 2)
    ax2.set_xticklabels(feature_names, rotation=15, fontsize=8)
    ax2.set_title("δ_i = β_i − β_global\n"
                  "Positive = dataset pulls coef UP\n"
                  "Negative = dataset pulls coef DOWN",
                  fontsize=9)
    ax2.set_ylabel("Deviation from global", fontsize=8)
    ax2.legend(fontsize=7)

    # Panel 2: Heatmap of δ_i
    # Rows = datasets, columns = features
    # Colour shows direction and magnitude of deviation
    ax3 = axes[2]
    delta_df = pd.DataFrame(delta_array,
                             index=labels,
                             columns=feature_names)
    sns.heatmap(delta_df, ax=ax3,
                annot=True, fmt=".4f",
                cmap="coolwarm", center=0,
                linewidths=0.5,
                cbar_kws={"label": "δ_i"})
    ax3.set_title("δ_i heatmap\n"
                  "Red = above global | Blue = below global\n"
                  "Large values → dataset differs from global model",
                  fontsize=9)

    plt.tight_layout()
    plt.savefig("03_global_deviation.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {"beta_global": beta_global,
            "beta_list":   beta_list,
            "delta_list":  delta_list}


# ==============================================================================
# SECTION 04 — POOLING COMPARISON
# ==============================================================================
"""
THEORY:
-------
When you have multiple related datasets, you face a fundamental choice:

(a) NO POOLING
    Fit a separate model f_i for each D_i.
    Each model only sees n_i samples.
    → High variance (small samples), ignores shared structure.
    → Correct if datasets are completely unrelated.

(b) FULL POOLING
    Combine D = ∪ D_i and fit one model f.
    → Maximum data, lowest variance.
    → Biased if datasets genuinely differ (assumes they are identical).

(c) PARTIAL POOLING
    Share information across datasets but allow variation.
    Implemented here as Ridge regression with a shared regularisation:
    the penalty shrinks β_i toward β_global.
    → Best of both worlds when datasets are related but not identical.
    → This is the practical implementation of the hierarchical model.

BIAS-VARIANCE TRADEOFF:
-----------------------
    No pooling:    low bias,  high variance  (overfits to each D_i)
    Full pooling:  high bias, low variance   (underfits dataset differences)
    Partial pool:  balanced   (shrinks toward global, allows deviation)

HOW TO READ THE RESULTS:
-------------------------
    Cross-partition MSE = train on D_i, test on D_j (i ≠ j)
    If full pool ≈ no pool → datasets are genuinely similar
    If full pool >> no pool → datasets differ, pooling hurts
    Partial pool should be ≤ both in cross-partition MSE
"""

def section_04_pooling(partitions, feature_names):
    print("=" * 70)
    print("SECTION 04 — POOLING COMPARISON")
    print("No pooling | Full pooling | Partial pooling")
    print("=" * 70)

    X_pool = np.vstack([p["X"] for p in partitions])
    y_pool = np.concatenate([p["y"] for p in partitions])

    # ── (a) No pooling: fit separate model per dataset ────────────────────
    no_pool_models = []
    for p in partitions:
        m = LinearRegression()
        m.fit(p["X"], p["y"])
        no_pool_models.append(m)

    # ── (b) Full pooling: one model on all data ───────────────────────────
    full_pool_model = LinearRegression()
    full_pool_model.fit(X_pool, y_pool)

    # ── (c) Partial pooling: Ridge per dataset ────────────────────────────
    # Ridge shrinks β_i toward zero (proxy for shrinking toward global)
    # Alpha controls how much shrinkage — higher = more pooling
    partial_pool_models = []
    for p in partitions:
        m = Ridge(alpha=1.0)
        m.fit(p["X"], p["y"])
        partial_pool_models.append(m)

    # ── Compute cross-partition MSE for each strategy ─────────────────────
    # For each strategy: train on D_i, test on D_j
    # Diagonal = in-sample, off-diagonal = out-of-sample
    def cross_mse_matrix(models, partitions):
        K   = len(partitions)
        mat = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                y_pred    = models[i].predict(partitions[j]["X"])
                mat[i, j] = mean_squared_error(partitions[j]["y"], y_pred)
        return pd.DataFrame(mat, index=labels, columns=labels)

    mse_no_pool      = cross_mse_matrix(no_pool_models, partitions)
    mse_partial_pool = cross_mse_matrix(partial_pool_models, partitions)

    # Full pooling: same model for all, so rows are identical
    mse_full_pool = pd.DataFrame(
        [[mean_squared_error(partitions[j]["y"],
                             full_pool_model.predict(partitions[j]["X"]))
          for j in range(K)]
         for _ in range(K)],
        index=labels, columns=labels)

    # Mean off-diagonal MSE (cross-partition generalisation)
    def mean_off_diag(mat):
        vals = mat.values.copy()
        np.fill_diagonal(vals, np.nan)
        return np.nanmean(vals)

    print(f"\n  Mean cross-partition MSE:")
    print(f"    No pooling:      {mean_off_diag(mse_no_pool):.4f}")
    print(f"    Full pooling:    {mean_off_diag(mse_full_pool):.4f}")
    print(f"    Partial pooling: {mean_off_diag(mse_partial_pool):.4f}")

    # ── Visualise ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Section 04 — Pooling Comparison\n"
                 "Row=train partition, Col=test partition  |  "
                 "Diagonal=in-sample, Off-diagonal=cross-partition",
                 fontsize=11, fontweight="bold")

    vmax = max(mse_no_pool.values.max(),
               mse_full_pool.values.max(),
               mse_partial_pool.values.max())

    for ax, mat, title in zip(
        axes[:3],
        [mse_no_pool, mse_full_pool, mse_partial_pool],
        ["(a) No pooling\nSeparate f_i per dataset",
         "(b) Full pooling\nOne f on all data",
         "(c) Partial pooling\nRidge — shrink toward global"]):
        sns.heatmap(mat.astype(float), ax=ax,
                    annot=True, fmt=".3f",
                    cmap="YlOrRd", vmin=0, vmax=vmax,
                    linewidths=0.5,
                    cbar_kws={"label": "MSE"})
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Test partition", fontsize=8)
        ax.set_ylabel("Train partition", fontsize=8)

    # Panel 3: Summary bar chart of mean cross-partition MSE
    ax4 = axes[3]
    strategies = ["No\npooling", "Full\npooling", "Partial\npooling"]
    mse_vals   = [mean_off_diag(mse_no_pool),
                  mean_off_diag(mse_full_pool),
                  mean_off_diag(mse_partial_pool)]
    bars = ax4.bar(strategies, mse_vals,
                   color=[PALETTE[0], PALETTE[1], PALETTE[2]],
                   edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, mse_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax4.set_title("Mean cross-partition MSE\n"
                  "Lower = better generalisation\n"
                  "across partitions",
                  fontsize=9)
    ax4.set_ylabel("Mean off-diagonal MSE", fontsize=8)

    plt.tight_layout()
    plt.savefig("04_pooling.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {"mse_no_pool":      mse_no_pool,
            "mse_full_pool":    mse_full_pool,
            "mse_partial_pool": mse_partial_pool}


# ==============================================================================
# SECTION 05 — DISTRIBUTION OVER COEFFICIENTS
# ==============================================================================
"""
THEORY:
-------
Instead of asking "what is β for dataset i?", ask:
    "What DISTRIBUTION does β follow across datasets?"

Treat the K fitted coefficient vectors as K samples from a distribution:

    β_j^(i)  ~  some distribution over i=1,...,K

For each feature j, compute:
    mean(β_j)  = average sensitivity across datasets
    std(β_j)   = how much that sensitivity varies
    range      = plausible range of the coefficient

INSIGHT:
--------
    Low std(β_j)  → all datasets agree on feature j's effect
                    → the relationship is UNIVERSAL
    High std(β_j) → feature j's effect is dataset-dependent
                    → the relationship is CONTEXT-SPECIFIC

This is the empirical version of the prior on θ in the hierarchical model.
It answers: "Which relationships are universal vs dataset-specific?"

EXTENSION:
----------
If you had many datasets (not just 4), you could:
    - Fit a parametric distribution to β_j^(i)
    - Use that distribution as a prior in a Bayesian model
    - Identify which features have multimodal β distributions
      (suggesting subgroups of datasets)
"""

def section_05_coef_distribution(partitions, feature_names):
    print("=" * 70)
    print("SECTION 05 — DISTRIBUTION OVER COEFFICIENTS")
    print("β^(i) as random variables across datasets")
    print("=" * 70)

    # ── Fit one model per partition and collect coefficients ──────────────
    beta_matrix = np.zeros((K, len(feature_names)))   # shape (K, p)

    for ki, p in enumerate(partitions):
        m = LinearRegression()
        m.fit(p["X"], p["y"])
        beta_matrix[ki] = m.coef_

    beta_df = pd.DataFrame(beta_matrix,
                            index=labels,
                            columns=feature_names)

    # ── Compute distribution statistics ───────────────────────────────────
    dist_stats = pd.DataFrame({
        "mean":  beta_df.mean(),
        "std":   beta_df.std(),
        "min":   beta_df.min(),
        "max":   beta_df.max(),
        "range": beta_df.max() - beta_df.min(),
        "cv":    (beta_df.std() / beta_df.mean().abs()).replace([np.inf], np.nan)
    })
    # cv = coefficient of variation = std / |mean|
    # high cv → high relative variability across datasets

    print("\n  Coefficient distribution across datasets:")
    print(dist_stats.round(4).to_string())
    print("\n  Interpretation:")
    for fname in feature_names:
        std_val = dist_stats.loc[fname, "std"]
        mean_val = dist_stats.loc[fname, "mean"]
        verdict = "UNIVERSAL" if std_val < 0.05 else "VARIABLE"
        print(f"    {fname:15s}  mean={mean_val:+.4f}  "
              f"std={std_val:.4f}  → {verdict}")

    # ── Visualise ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Section 05 — Distribution Over Coefficients\n"
                 "β^(i) treated as samples from a distribution across datasets",
                 fontsize=11, fontweight="bold")

    # Panel 0: Strip plot — each dot = one dataset's coefficient
    # Shows the full distribution of β_j across datasets
    ax = axes[0]
    for fi, fname in enumerate(feature_names):
        for ki in range(K):
            ax.scatter(fi, beta_matrix[ki, fi],
                       color=PALETTE[ki], s=80, zorder=3,
                       label=labels[ki] if fi == 0 else "")
        # Overlay mean ± std
        mean_val = beta_df[fname].mean()
        std_val  = beta_df[fname].std()
        ax.errorbar(fi, mean_val, yerr=std_val,
                    fmt="D", color="black", markersize=6,
                    capsize=5, linewidth=2, zorder=4,
                    label="mean ± std" if fi == 0 else "")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=15, fontsize=8)
    ax.set_title("β^(i) per feature\n"
                 "Dots = individual datasets\n"
                 "Diamond = mean ± std",
                 fontsize=9)
    ax.set_ylabel("Coefficient value", fontsize=8)
    ax.legend(fontsize=7)

    # Panel 1: std(β_j) bar chart
    # Directly shows which features have UNIVERSAL vs VARIABLE effects
    ax2 = axes[1]
    std_vals = beta_df.std()
    bars = ax2.bar(feature_names, std_vals,
                   color=[PALETTE[i] for i in range(len(feature_names))],
                   edgecolor="white")
    for bar, val in zip(bars, std_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_title("std(β_j) across datasets\n"
                  "Low  → universal relationship\n"
                  "High → dataset-specific relationship",
                  fontsize=9)
    ax2.set_ylabel("Standard deviation of β", fontsize=8)
    ax2.set_xticklabels(feature_names, rotation=15, fontsize=8)

    # Panel 2: Heatmap of β_matrix
    # Rows = datasets, columns = features
    # Colour shows coefficient value — look for rows that stand out
    ax3 = axes[2]
    sns.heatmap(beta_df, ax=ax3,
                annot=True, fmt=".4f",
                cmap="coolwarm", center=0,
                linewidths=0.5,
                cbar_kws={"label": "β value"})
    ax3.set_title("β^(i) heatmap\n"
                  "Rows = datasets | Cols = features\n"
                  "Consistent colour → universal effect",
                  fontsize=9)

    plt.tight_layout()
    plt.savefig("05_coef_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {"beta_df": beta_df, "dist_stats": dist_stats}


# ==============================================================================
# SECTION 06 — FUNCTIONAL ALIGNMENT
# ==============================================================================
"""
THEORY:
-------
Instead of comparing model PARAMETERS β_i (which are abstract numbers),
compare what the models actually DO on a shared input grid.

Construct a prediction vector for each dataset:

    F_i = [f_i(x^(1)), f_i(x^(2)), ..., f_i(x^(m))]

Each dataset is now represented as a VECTOR OF PREDICTIONS
over a shared domain.

Then:
    - Plot f_i(x) curves to see where models agree/disagree
    - Run PCA on {F_i} to find the main axes of functional variation
    - Compute functional distance: mean( (f_i(x) - f_j(x))² )

WHY THIS IS BETTER THAN COMPARING β_i DIRECTLY:
-------------------------------------------------
    Two models can have very different β_i but behave almost identically
    if the features are correlated.

    Two models can have similar β_i but diverge strongly in a specific
    region of the input space.

    Functional alignment captures BEHAVIOUR, not just parameters.

SHARED INPUT GRID:
------------------
    We vary one feature at a time (holding others at their mean)
    to produce interpretable 1D slices of the model's behaviour.
    This is the standard "partial dependence" approach.
"""

def section_06_functional_alignment(partitions, feature_names):
    print("=" * 70)
    print("SECTION 06 — FUNCTIONAL ALIGNMENT")
    print("F_i = [f_i(x^1), ..., f_i(x^m)]  — compare model behaviour")
    print("=" * 70)

    # ── Step 1: Fit one model per partition ───────────────────────────────
    models = []
    for p in partitions:
        m = LinearRegression()
        m.fit(p["X"], p["y"])
        models.append(m)

    # ── Step 2: Build shared reference grid ───────────────────────────────
    # Use the pooled data as the shared domain
    X_pool   = np.vstack([p["X"] for p in partitions])
    X_means  = X_pool.mean(axis=0)   # mean of each feature (for holding fixed)
    m_grid   = 100                   # number of grid points per feature

    # ── Step 3: Compute prediction vectors F_i on pooled grid ─────────────
    # F_i = model i's predictions on ALL observed data points
    # This gives a (n_total,) vector per model
    F_matrix = np.zeros((K, len(X_pool)))   # shape (K, n_total)
    for ki, model in enumerate(models):
        F_matrix[ki] = model.predict(X_pool)

    # ── Step 4: Functional distance matrix ────────────────────────────────
    # mean( (f_i(x) - f_j(x))² ) over the shared grid
    func_dist = pd.DataFrame(np.zeros((K, K)), index=labels, columns=labels)
    for i in range(K):
        for j in range(K):
            func_dist.iloc[i, j] = np.mean((F_matrix[i] - F_matrix[j]) ** 2)

    print("\n  Functional distance  mean( (f_i(x) - f_j(x))² ):")
    print(func_dist.round(6).to_string())

    # ── Step 5: PCA on F_matrix ───────────────────────────────────────────
    # Each row = one dataset's prediction vector
    # PCA finds the main axes of functional variation across datasets
    pca    = PCA(n_components=2)
    F_pca  = pca.fit_transform(F_matrix)   # shape (K, 2)

    # ── Step 6: Partial dependence curves ────────────────────────────────
    # For each feature, vary it across its range while holding others fixed
    # This shows WHERE in the input space models agree or diverge
    pd_curves = {}   # {feature_name: (x_grid, {label: predictions})}
    for fi, fname in enumerate(feature_names):
        x_range = np.linspace(X_pool[:, fi].min(),
                               X_pool[:, fi].max(), m_grid)
        X_grid  = np.tile(X_means, (m_grid, 1))   # all features at mean
        X_grid[:, fi] = x_range                    # vary feature fi

        preds = {}
        for ki, model in enumerate(models):
            preds[labels[ki]] = model.predict(X_grid)
        pd_curves[fname] = (x_range, preds)

    # ── Visualise ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Section 06 — Functional Alignment\n"
                 "Row 0: Partial dependence curves (where do models agree?)\n"
                 "Row 1: PCA of F_i vectors | Functional distance heatmap",
                 fontsize=11, fontweight="bold")

    gs = gridspec.GridSpec(2, len(feature_names) + 1,
                           figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: Partial dependence curves — one panel per feature
    # Overlapping lines → models agree in this region
    # Diverging lines   → models disagree — dataset-specific behaviour
    for fi, fname in enumerate(feature_names):
        ax = fig.add_subplot(gs[0, fi])
        x_range, preds = pd_curves[fname]
        for ki, label in enumerate(labels):
            ax.plot(x_range, preds[label],
                    color=PALETTE[ki], linewidth=2, label=label)
        ax.set_title(f"f_i(x) varying {fname}\n"
                     f"(others held at mean)",
                     fontsize=8)
        ax.set_xlabel(fname, fontsize=7)
        ax.set_ylabel("Predicted y", fontsize=7)
        if fi == 0:
            ax.legend(fontsize=6)

    # Row 1, col 0–1: PCA of F_i vectors
    # Each point = one dataset's FUNCTION (not parameters)
    # Distance between points = functional dissimilarity
    ax_pca = fig.add_subplot(gs[1, :2])
    ax_pca.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax_pca.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    for ki, label in enumerate(labels):
        ax_pca.scatter(F_pca[ki, 0], F_pca[ki, 1],
                       color=PALETTE[ki], s=150, zorder=3)
        ax_pca.annotate(f"  {label}\n  ({F_pca[ki,0]:.3f}, {F_pca[ki,1]:.3f})",
                        (F_pca[ki, 0], F_pca[ki, 1]),
                        fontsize=8, color=PALETTE[ki], fontweight="bold")
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8)
    ax_pca.set_title("PCA of prediction vectors F_i\n"
                     "Distance = functional dissimilarity\n"
                     "Clustered = models behave similarly",
                     fontsize=9)

    # Row 1, col 2–3: Functional distance heatmap
    ax_heat = fig.add_subplot(gs[1, 2:])
    sns.heatmap(func_dist.astype(float),
                ax=ax_heat,
                annot=True, fmt=".6f",
                cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "mean( (f_i - f_j)² )"})
    ax_heat.set_title("Functional distance matrix\n"
                      "mean( (f_i(x) - f_j(x))² ) over shared grid\n"
                      "0 = models behave identically",
                      fontsize=9)

    plt.savefig("06_functional_alignment.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {"F_matrix":   F_matrix,
            "F_pca":      F_pca,
            "func_dist":  func_dist,
            "pd_curves":  pd_curves}


# ==============================================================================
# MAIN — RUN ALL SECTIONS
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("POPULATION OF DATASETS — STRUCTURAL ANALYSIS")
    print("Moving from pairwise comparison to modelling a distribution")
    print("over datasets")
    print("=" * 70 + "\n")

    # ── Section 02: What is the average dataset? ──────────────────────────
    results_02 = section_02_hierarchical(partitions, feature_names)

    # ── Section 03: Is there one global formula with per-dataset tweaks? ──
    results_03 = section_03_global_deviation(partitions, feature_names)

    # ── Section 04: Should I pool the data? ───────────────────────────────
    results_04 = section_04_pooling(partitions, feature_names)

    # ── Section 05: Which relationships are universal? ────────────────────
    results_05 = section_05_coef_distribution(partitions, feature_names)

    # ── Section 06: Where do models agree and disagree? ───────────────────
    results_06 = section_06_functional_alignment(partitions, feature_names)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — WHAT EACH SECTION ANSWERED")
    print("=" * 70)
    print("""
  02  Hierarchical model
      μ_0  = the average dataset (population mean)
      Σ_0  = which features vary most across datasets
      → Tells you what a "typical" dataset looks like

  03  Global model + deviation
      β_global = shared regression structure
      δ_i      = how much each dataset deviates from it
      → Tells you whether one formula fits all datasets

  04  Pooling comparison
      No / Full / Partial pooling MSE comparison
      → Tells you whether combining datasets helps or hurts

  05  Distribution over coefficients
      std(β_j) per feature across datasets
      → Tells you which feature relationships are universal
        vs dataset-specific

  06  Functional alignment
      F_i = prediction vectors on shared grid
      PCA + functional distance matrix
      → Tells you WHERE in the input space models agree/disagree
    """)
