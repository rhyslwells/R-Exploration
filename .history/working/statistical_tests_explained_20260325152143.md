# Statistical Tests Reference Guide

## Purpose
This document explains the logic, interpretation, and use cases for statistical tests. The tests are organized by analytical goal. Implementation examples refer to `0-example1.R` for context, but the concepts apply broadly to any dataset.



---

## 1. CORRELATION & REGRESSION ANALYSIS

### Correlation Matrix
**Purpose:** Identify which pairs of numeric variables are linearly related.

**Logic:**
- Calculates Pearson correlation coefficient between each pair of variables
- Returns values from -1 to +1
  - **+1 to 0.7:** Strong positive relationship
  - **0.7 to 0.3:** Moderate to weak positive relationship
  - **-0.3 to -0.7:** Moderate to weak negative relationship
  - **-0.7 to -1:** Strong negative relationship
  - **~0:** No linear relationship

**Interpretation:** High correlation suggests variables change together, but does NOT imply causation. Variables may be correlated due to a third factor or random chance.

**Implementation example:**
```r
# Calculate correlation matrix for numeric variables
correlation_matrix <- cor(data[, c("var1", "var2", "var3")], use = "complete.obs")

# Visualize as heatmap
heatmap(correlation_matrix)
```

**When to use:** Exploratory analysis to discover which variables co-vary.

---

### Simple Linear Regression
**Purpose:** Quantify the linear relationship between one predictor variable (X) and one response variable (Y).

**Logic:**
- Fits a line: Y = a + b*X
  - **Intercept (a):** Y value when X = 0
  - **Slope (b):** Change in Y per unit change in X
- Measures goodness of fit with **R²** (proportion of Y's variance explained by X)

**R² interpretation:**
- **0.9-1.0:** Excellent fit, X explains ~90-100% of Y's variation
- **0.5-0.9:** Good fit
- **0.2-0.5:** Poor fit, other factors important
- **<0.2:** Very weak relationship

**P-value interpretation:**
- **p < 0.05:** Relationship is statistically significant (unlikely due to random chance)
- **p > 0.05:** No significant linear relationship detected

**Implementation example:**
```r
# Fit simple linear regression
model <- lm(Y ~ X, data = mydata)
summary(model)  # Shows slope, intercept, R², p-value
```

**When to use:** When you want to predict or understand how one variable affects another.

---

### Multiple Linear Regression
**Purpose:** Test if multiple predictor variables together explain variation in a response variable, and determine which predictors are most important.

**Logic:**
- Extends simple regression: Y = a + b₁*X₁ + b₂*X₂ + b₃*X₃ + ...
- Each slope (b) shows the change in Y per unit of that X, holding other X variables constant (partial effect)
- **Overall R²:** Proportion of Y's variance explained by ALL X variables combined
- **Individual p-values:** Test if each predictor is significantly useful

**Interpretation:**
- Compare slopes to see relative importance of predictors
- Non-significant p-value for a predictor → that variable doesn't add meaningful information
- Higher overall R² than simple regression → added predictors improve the model

**Implementation example:**
```r
# Multiple regression with 3 predictors
model <- lm(Y ~ X1 + X2 + X3, data = mydata)
summary(model)  # Shows each coefficient, its p-value, and overall R²
```

**When to use:** Exploratory analysis to find most important factors; adjust for confounding variables; test competing hypotheses about what drives outcomes.

---


## 2. GROUP COMPARISONS (ANOVA)

### One-way ANOVA: weight by maturity stage
**What we're looking for:**
- Are mean weights significantly DIFFERENT across the 5 maturity stages?
- **Null hypothesis (H₀):** All maturity stages have the same average weight
- **Alternative hypothesis:** At least one maturity stage has a different average weight
- **P-value < 0.05 means:** Reject H₀ - maturity stages DO differ significantly in weight

**Key insights:** Does reproductive development correlate with body size?

---

### One-way ANOVA: DML by maturity stage
**What we're looking for:**
- Are squids at different maturity stages also at different sizes?
- Helps separate growth effects from reproductive investment effects

**Key insights:** Are larger squids more reproductively mature, or is maturity independent of size?

---

### Tukey HSD (Honestly Significant Difference) Test
**What we're looking for:**
- ANOVA tells us IF there are differences; Tukey tells us WHICH groups differ
- Compares all pairs of maturity stages (1 vs 2, 1 vs 3, etc.)
- Adjusts for multiple comparisons to avoid false positives
- P-values < 0.05 indicate significant pairwise differences

**Key insights:** Which specific maturity stages differ from each other?

---

### Pairwise t-tests with Bonferroni correction
**What we're looking for:**
- Alternative method to Tukey for identifying group differences
- Bonferroni correction is more conservative (stricter p-value threshold)
- Reduces risk of Type I error (false positives)

**Key insights:** Confirms which maturity stage pairs differ significantly (conservative approach)

---

## 3. DISTRIBUTIONS & ASSUMPTIONS TESTING

### Shapiro-Wilk Test for Normality
**What we're looking for:**
- Tests if weight/DML values follow a normal (bell-curve) distribution
- **Null hypothesis:** Data IS normally distributed
- **P-value < 0.05 means:** Data is NOT normally distributed (reject normality)
- **P-value > 0.05 means:** Data distribution is consistent with normal (don't reject)

**Why it matters:** Many statistical tests assume normality. If violated, results may be unreliable.

**Key insights:** Can we trust parametric tests (ANOVA, t-tests, regression)? Or do we need non-parametric alternatives?

---

### Density Plot with Normal Curve Overlay
**What we're looking for:**
- Visual inspection of whether actual data distribution matches a theoretical normal curve
- See if there are multiple peaks (bimodal) or skewness
- Red dashed line shows what perfect normality would look like

**Key insights:** Are squids a single population or multiple groups?

---

### Q-Q Plot (Quantile-Quantile Plot)
**What we're looking for:**
- Points should fall on the red diagonal line if data is normally distributed
- Points deviating from the line indicate non-normality
- Deviations at the tails show non-normal behavior in extreme values

**Key insights:** Where specifically does normality break down?

---

## 4. TEMPORAL ANALYSIS

### T-test: weight comparison between years (1989 vs 1990)
**What we're looking for:**
- Are mean weights significantly different between the two years?
- **P-value < 0.05:** Weights differ between years (significant change)
- **P-value > 0.05:** Both years have similar weight distributions

**Key insights:** Did environmental or population conditions change between years?

---

### ANOVA: weight by month (seasonal patterns)
**What we're looking for:**
- Do squids collected in different months have different weights?
- Tests for seasonal variation in seafood quality/nutrition availability
- **Significant p-value:** Some months differ from others

**Key insights:** Is there a seasonal pattern to squid body condition?

---

## 5. REPRODUCTIVE INVESTMENT ANALYSIS

### Allometric Regression: ovary.weight ~ DML
**What we're looking for:**
- How does reproductive investment scale with body size?
- **Positive slope:** Larger squids invest more in reproduction (expected)
- **Strength of relationship:** Is ovary investment tightly linked to body size?

**Key insights:** Do all squids of the same size have similar reproductive output?

---

### Reproductive Investment Ratio (ovary.weight / body.weight)
**What we're looking for:**
- What proportion of body weight is allocated to reproduction at each maturity stage?
- **Higher ratio:** More reproductive effort
- **Pattern across stages:** Does investment increase with maturity?

**Key insights:** How do reproductive priorities change during development?

---

### ANOVA: Investment Ratio by maturity stage
**What we're looking for:**
- Do maturity stages differ in reproductive effort allocation?
- **Significant p-value:** Different stages prioritize reproduction differently

**Key insights:** Is there a threshold effect (sudden shift) or gradual change in reproductive investment?

---

### Correlation: nidamental gland weight vs ovary weight
**What we're looking for:**
- Both are reproductive organs in females
- Are they developmentally linked? (Do they develop together?)
- **Strong correlation:** Coordinated development
- **Weak correlation:** Independent development

**Key insights:** Is reproductive system development synchronized?

---

## 6. MULTIVARIATE ANALYSIS - PCA (Principal Component Analysis)

### What PCA is looking for overall:
- Reduces many correlated measurements to a few uncorrelated components
- Reveals hidden patterns and groupings in the data
- Answers: What are the main ways squids vary?

---

### Scree Plot
**What we're looking for:**
- How many components are needed to explain most variation?
- Look for an "elbow" - point where adding more components doesn't help much
- Each bar = variance explained by that principal component

**Key insights:** Are squids varying along 1 main dimension, 2 dimensions, or more?

---

### Cumulative Variance Explained Plot
**What we're looking for:**
- How many PCs do we need to explain 80% or 90% of variation?
- Red dashed lines at 0.8 and 0.9 show common thresholds

**Key insights:** Is squid variation simple (described by 1-2 factors) or complex?

---

### Biplot
**What we're looking for:**
- Shows which variables load heavily on which principal components
- Arrows pointing same direction = correlated variables
- Perpendicular arrows = unrelated variables

**Key insights:** Which measurements go together? What biological patterns exist?

---

## 7. CLUSTERING ANALYSIS

### Hierarchical Clustering Dendrogram
**What we're looking for:**
- Groups squids into natural clusters based on all measurements
- Height of branches shows how different clusters are
- Squids in same cluster = similar overall body composition

**Key insights:** Are there distinct squid types, or is variation continuous? Can we identify subpopulations?

---

### Cluster Assignment Analysis (3 clusters)
**What we're looking for:**
- Characterize the 3 identified clusters
- Are they different sizes? Different maturity stages?
- Can we describe ecological or biological meaning of clusters?

**Key insights:** What do the clusters represent - age groups, condition classes, or population substructure?

---

## 8. CHI-SQUARE TEST: Year vs Maturity Stage

### What we're looking for:
- Tests if **proportions** of maturity stages differ between years
- Different from t-test/ANOVA which test **means**
- **Null hypothesis:** Maturity stage distribution is independent of year
- **P-value < 0.05:** Year and maturity stage are associated (not independent)

**Key insights:** Did the population age structure change between years? Were different cohorts collected?

---

## Summary: What Questions Are We Answering?

| Question | Test(s) |
|----------|---------|
| Which variables are related? | Correlation, regression |
| Does maturity stage affect body size? | ANOVA + Tukey |
| Are our findings robust statistically? | P-values, assumptions testing |
| Is there seasonal variation? | Temporal ANOVA |
| How much do squids invest in reproduction? | Allometric analysis, investment ratios |
| Can we simplify the data? | PCA |
| Are there distinct squid groups? | Clustering, dendrogram |
| Did population composition change in time? | Chi-square, t-tests |

---

## Red Flags to Look For

- **Non-normal distributions** → May need non-parametric alternatives
- **Low R² values** → Other factors are important
- **Non-significant p-values** → No strong evidence of differences
- **Outliers in plots** → May indicate data entry errors or unusual specimens
- **Few clusters in dendrogram** → Simple, structured population
- **Many clusters needed** → Complex variation patterns
