Given:

datasets are independent samples of the same underlying process

you want a single interpretable global model


the problem reduces to estimating a shared functional relationship while accounting for sampling variability across datasets.

There are three principled approaches, ordered by increasing sophistication.


---

# 1. Full pooling (baseline)

Concatenate all datasets:

D = \bigcup_{i=1}^n D_i

Fit a single model:

y = f(x; \theta)

Example (linear):

y = \beta_0 + \beta_1 x_1 + \cdots + \beta_4 x_4

Interpretation:

$\theta$ estimates the true underlying process

variability across datasets is treated as noise


When this works:

datasets are genuinely i.i.d.

no systematic differences between datasets


Limitation:

cannot detect or quantify between-dataset variation



---

# 2. Partial pooling (recommended)

Assume:

y_{ij} = f(x_{ij}; \theta) + \epsilon_{ij}

but allow small dataset-specific deviations:

\theta_i = \theta + \delta_i, \quad \delta_i \sim \mathcal{N}(0, \Sigma_\delta)

Linear example:

y_{ij} = (\beta + u_i)^T x_{ij} + \epsilon_{ij}

where:

$\beta$ = global coefficients (what you want)

$u_i$ = dataset-specific adjustments


Outcome:

$\beta$ → interpretable global model

$\Sigma_\delta$ → how much datasets vary


This is a mixed-effects model.


---

3. Symbolic regression on pooled data (interpretable formula)

If interpretability is a priority beyond linearity:

Fit:

y = f(x_1, x_2, x_3, x_4)

using symbolic regression on pooled data.

Example output:

y = a \cdot x_1 + b \cdot \log(x_2) + c \cdot x_3^2

Then validate across datasets:

evaluate error per dataset $D_i$

check stability of the formula



---

4. Critical step: validate the “same process” assumption

Even if assumed, you should test it.

Procedure:

1. Fit global model $f$


2. Compute per-dataset error:



E_i = \frac{1}{|D_i|} \sum (y - f(x))^2

3. Analyse ${E_i}$:



low variance → assumption holds

high variance → hidden heterogeneity



---

5. Strengthen interpretability

To ensure the global model is meaningful:

(a) Normalisation

Ensure all datasets share:

same units

comparable ranges


(b) Feature stability

Check that relationships hold across datasets:

sign of coefficients consistent

similar marginal effects


(c) Residual structure

If residuals cluster by dataset: → missing variable or structural difference


---

# 6. Recommended workflow

Step 1: Pool data

D = \bigcup_i D_i

Step 2: Fit interpretable model

Start simple:

linear / regularised linear

Step 3: Diagnose per dataset

error $E_i$

residual distribution

coefficient stability (via refits or bootstrap)

Step 4: Upgrade if needed

If heterogeneity appears: → move to mixed-effects model


---

# Concrete example


Variables:

$T, F, P \rightarrow C$


Global model:

C = \beta_0 + \beta_1 T + \beta_2 F + \beta_3 P

Interpretation:

$\beta_1$: average temperature sensitivity across all datasets

$\beta_2$: flow effect

$\beta_3$: pressure effect


Validation:

compute $E_i$ per dataset

plot residuals vs dataset index


If stable: → model represents the shared process

If not: → introduce:

\beta_1^{(i)} = \beta_1 + u_{1i}


---

Key point

Since your datasets are independent samples of the same process, the most defensible interpretation is:

\text{Estimate } f \text{ such that } y = f(x) \text{ captures the underlying mechanism}

Then treat variation across datasets as:

noise (if small)

structured deviation (if not)
