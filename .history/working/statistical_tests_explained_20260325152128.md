# Statistical Tests Reference Guide

## Purpose
This document explains the logic, interpretation, and use cases for statistical tests. The tests are organized by analytical goal. Implementation examples refer to `0-example1.R` for context, but the concepts apply broadly to any dataset.



---

## 1. CORRELATION & REGRESSION ANALYSIS

### Correlation Matrix
**What we're looking for:** 
- Identifies which variables are strongly related to each other
- Correlation values range from -1 to +1
- Values close to +1 indicate positive relationships (as one variable increases, so does the other)
- Values close to -1 indicate negative relationships (as one increases, the other decreases)
- Values close to 0 indicate no linear relationship

**Key insights:** Which body measurements change together? For example, do heavier squids have longer body lengths?

---

### Simple Linear Regression (weight ~ DML)
**What we're looking for:**
- Quantifies the relationship between body length (DML) and weight
- **R-squared:** How much of weight variation is explained by DML? (0-1 scale, higher is better)
- **Slope (coefficient):** How many grams of weight increase per mm of DML?
- **P-value:** Is this relationship statistically significant (p < 0.05)?

**Key insights:** Can we predict squid weight from their body length? How strong is this relationship?

---

### Multiple Linear Regression (weight ~ DML + ovary.weight + maturity.stage)
**What we're looking for:**
- Tests if reproductive stage and ovary development help explain weight BEYOND just body length
- Are there remaining unexplained differences after accounting for physical size?
- Which predictors are actually important?

**Key insights:** What factors matter most for explaining squid weight?

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
