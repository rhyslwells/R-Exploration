# Dataset

Use a datset from there a sklearn dataset we can use as th global dataset which we can partition (non time dimension). We will take D_i as equal in size random samples. something with 4 variables

sampels will have the same variabeles and scales

# Compare datasets:

Ways to compare datasets. The right approach depends on what you mean by “compare”: distributional similarity, structural similarity, or functional behaviour.

## 1. Compare via summary parameters (descriptive statistics)

For each dataset $D_i$, compute a vector of statistics:

Location: mean $\mu_i$, median

Spread: variance $\sigma_i^2$, IQR

Shape: skewness, kurtosis

Size: $n_i$

Domain-specific metrics (e.g. energy consumption per degree-day)

This gives a feature vector:

\phi(D_i) = [\mu_i, \sigma_i^2, \text{skew}_i, \dots]

Then compare datasets via:

Distance: $|\phi(D_i) - \phi(D_j)|$

Clustering (group similar datasets)

PCA to understand variation across datasets


This is simple and scalable, but loses structure.


---

## 2. Compare distributions directly

Treat each dataset as a sample from a distribution $P_i$.

Methods:

Kolmogorov–Smirnov test (1D)

Wasserstein distance:

W(P_i, P_j)

D_{KL}(P_i \parallel P_j)

These retain more information than summary stats.

For multivariate datasets:

Maximum Mean Discrepancy (MMD)

Energy distance


This is useful when datasets differ in shape rather than just mean/variance.


---

## 3. Compare via fitted models / functions

This aligns with your idea of “overarching formula”.

Approach:

Fit a model $f_i(x)$ to each dataset:

Linear regression

Then compare:

(a) Parameter comparison

If models are parametric:

f_i(x) = \beta_i^T x

(b) Functional similarity

Compare outputs over a shared domain:

\int (f_i(x) - f_j(x))^2 dx

(c) Cross-evaluation

Train on $D_i$, test on $D_j$:

If performance holds → similar generating process

If it degrades → structural difference



## Follow-up directions

Are your datasets aligned on the same variables and scale?

Are you more interested in detecting differences or grouping similar datasets?