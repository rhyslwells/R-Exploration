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


## 2. GROUP COMPARISONS (ANOVA & Post-hoc Tests)

### One-way ANOVA (Analysis of Variance)
**Purpose:** Test if the means of a continuous response variable differ significantly across groups.

**Logic:**
- Tests the null hypothesis: **H₀** = all group means are equal
- Compares between-group variation to within-group variation
- If between-group variance is much larger than within-group variance, groups likely differ
- Produces an **F-statistic** and **p-value**

**Null Hypothesis (H₀):** All group means are equal  
**Alternative Hypothesis (H₁):** At least one group mean differs

**P-value interpretation:**
- **p < 0.05:** Reject H₀ → Groups differ significantly (beyond expected random variation)
- **p > 0.05:** Fail to reject H₀ → Insufficient evidence that groups differ

**Assumptions:**
- Approximately normal distributions within each group
- Roughly equal variances across groups
- Observations are independent

**Implementation example:**
```r
# ANOVA: test if response variable differs across groups
anova_result <- aov(response_variable ~ group, data = mydata)
summary(anova_result)
```

**When to use:** Compare means across 3+ categories; first step before identifying which specific groups differ.

**Limitation:** ANOVA only tells you IF groups differ, not WHICH groups differ. Use post-hoc tests for specific comparisons.

---

### Tukey HSD (Honestly Significant Difference) Test
**Purpose:** After ANOVA, identify which specific group pairs differ significantly.

**Logic:**
- Performs pairwise comparisons between all group pairs
- Adjusts p-values to control for multiple comparisons (reduces false positives)
- Controls **Family-Wise Error Rate (FWER)** at α = 0.05

**P-value interpretation:**
- **p < 0.05:** Those two groups differ significantly
- **p > 0.05:** No significant difference between those two groups

**Implementation example:**
```r
# Fit ANOVA model first
model <- aov(response_variable ~ group, data = mydata)

# Apply Tukey HSD
tukey_result <- TukeyHSD(model)
print(tukey_result)  # Shows p-values for all pairwise comparisons
```

**When to use:** After ANOVA shows significance, to determine which specific groups/pairs drive the overall difference.

**Advantage over multiple t-tests:** Automatically corrects for doing many comparisons simultaneously.

---

### Pairwise t-tests with Bonferroni Correction
**Purpose:** Alternative to Tukey for identifying group differences; more conservative.

**Logic:**
- Performs independent t-tests for each pair of groups
- Bonferroni correction divides α (0.05) by number of comparisons
  - If 10 comparisons: effective α = 0.05/10 = 0.005 (much stricter)
- More conservative → fewer false positives but may miss real differences

**Implementation example:**
```r
# Pairwise t-tests with Bonferroni correction
pairwise_result <- pairwise.t.test(response_variable, group, 
                                    p.adjust.method = "bonferroni")
print(pairwise_result)
```

**When to use:** When you want a very conservative approach; when you specifically want to control false positives.

---


## 3. DISTRIBUTIONS & ASSUMPTIONS TESTING

### Shapiro-Wilk Test for Normality
**Purpose:** Test whether a sample of data follows a normal distribution.

**Logic:**
- Compares observed data distribution to theoretical normal distribution
- Produces **W-statistic** and **p-value**
- W close to 1 → likely normal; W close to 0 → likely non-normal

**Null Hypothesis (H₀):** Data IS normally distributed  
**Alternative Hypothesis (H₁):** Data is NOT normally distributed

**P-value interpretation:**
- **p < 0.05:** Reject H₀ → Data is significantly non-normal
- **p > 0.05:** Fail to reject H₀ → Data distribution consistent with normal

**Why it matters:** Many statistical tests (ANOVA, regression, t-tests) assume normality. Violation can make results unreliable, especially with small samples.

**Implementation example:**
```r
# Test for normality
shapiro_result <- shapiro.test(mydata$variable)
print(shapiro_result)  # Shows W-statistic and p-value
```

**Limitations:**
- Very sensitive to sample size (large N will almost always reject H₀)
- Minor deviations from normality may not matter in practice
- Visual inspection often more informative than the test

---

### Density Plot with Normal Curve Overlay
**Purpose:** Visually compare observed data distribution to theoretical normal distribution.

**What to look for:**
- **Perfect normal:** Bell curve centered at mean, symmetric tails
- **Bimodal:** Two peaks → possible distinct subgroups
- **Skewed right:** Long tail extending right → right-skewed distribution
- **Skewed left:** Long tail extending left → left-skewed distribution
- **Flat/uniform:** No clear peak → data spread evenly across range

**Implementation example:**
```r
# Density plot with normal curve overlay
plot(density(mydata$variable), main = "Density Plot")
curve(dnorm(x, mean = mean(mydata$variable), sd = sd(mydata$variable)), 
      col = "red", add = TRUE, lty = 2)
```

**When to use:** Exploratory inspection; better than statistical tests for practical assessment.

---

### Q-Q Plot (Quantile-Quantile Plot)
**Purpose:** Visually assess normality by comparing observed data quantiles to theoretical normal quantiles.

**What to look for:**
- **Perfect normal:** Points fall on diagonal red line
- **Deviation at tails:** Points deviate from line at top/bottom → non-normal extremes
- **Systematic curves:** S-shaped → data more extreme than normal; reverse S → data less extreme
- **Scattered points:** Random deviations from line → approximately normal

**Implementation example:**
```r
# Q-Q plot
qqnorm(mydata$variable, main = "Q-Q Plot")
qqline(mydata$variable, col = "red")
```

**Advantage over Shapiro-Wilk test:** Shows WHERE and HOW data deviates from normality (more informative).

---


## 4. COMPARING GROUPS ACROSS CATEGORIES

### Independent Samples t-test
**Purpose:** Compare means of a continuous variable between exactly two groups.

**Logic:**
- Tests whether two independent groups have significantly different means
- Produces **t-statistic** and **p-value**
- Assumes both groups approximately normally distributed with roughly equal variances

**Null Hypothesis (H₀):** The two group means are equal  
**Alternative Hypothesis (H₁):** The two group means differ

**P-value interpretation:**
- **p < 0.05:** Groups have significantly different means
- **p > 0.05:** Insufficient evidence that means differ

**Implementation example:**
```r
# Compare means between two groups
group1 <- mydata[mydata$category == "A", "response_var"]
group2 <- mydata[mydata$category == "B", "response_var"]

t_result <- t.test(group1, group2)
print(t_result)  # Shows t-statistic, p-value, and 95% CI
```

**When to use:** Comparing before/after, treatment vs control, or any two-group comparison.

**Related note:** ANOVA is the generalization for 3+ groups.

---

### ANOVA Across Multiple Categories
**Purpose:** Compare means across multiple categorical levels of one variable.

**Implementation example:**
```r
# ANOVA: test if response differs across months, years, or other categories
anova_result <- aov(response_variable ~ category, data = mydata)
summary(anova_result)
```

**Interpretation:** Same as ANOVA section (Section 2).

---


## 5. ANALYZING RELATIONSHIPS & PROPORTIONS

### Linear Regression Relationships
**Purpose:** Quantify how one variable scales with another (allometry, growth relationships, etc.).

**Logic:**
- Fits Y = a + b*X  
- Slope (b) shows rate of change
- R² shows how much variation is explained

**Implementation example:**
```r
# Regression for scaling relationships
model <- lm(dependent_variable ~ independent_variable, data = mydata)
summary(model)

# Visualize
plot(data$x, data$y)
abline(model, col = "red")  # Add regression line
```

**Interpretation:**
- **Positive slope:** Positive relationship (both increase together)
- **Negative slope:** Negative relationship (inverse pattern)
- **Weak slope:** Minimal change in Y per unit X

---

### Proportion/Ratio Analysis
**Purpose:** Compare how much of a whole is allocated to different components, typically by comparing ratios across groups.

**Logic:**
- Create ratio: Component / Total
- Compare ratios across groups using ANOVA or visualization
- Tests whether allocation priorities differ

**Implementation example:**
```r
# Calculate ratio
data$proportion_ratio <- data$component / data$total

# Compare ratios across groups
boxplot(proportion_ratio ~ group, data = mydata)

# Statistical test
anova_result <- aov(proportion_ratio ~ group, data = mydata)
summary(anova_result)
```

**When to use:** Compare how resources/effort are allocated; identify if priorities shift across conditions.

---

### Correlation Between Related Variables
**Purpose:** Test whether two related variables (e.g., two components of a system) co-vary or develop independently.

**Logic:**
- Strong correlation → coordinated development/behavior
- Weak correlation → independent variation

**Implementation example:**
```r
# Calculate correlation coefficient
cor_value <- cor(data$variable1, data$variable2)

# Visual check
plot(data$variable1, data$variable2)
abline(lm(variable2 ~ variable1, data = data), col = "blue")
```

**When to use:** Understand whether processes are linked or independent.

---


## 6. MULTIVARIATE ANALYSIS - PCA (Principal Component Analysis)

### Purpose of PCA
**Overall goal:** Reduce many correlated variables into a few uncorrelated components while retaining as much variation as possible.

**When to use:** 
- Dataset has many intercorrelated measurements
- Want to identify main patterns/axes of variation
- Need to simplify dataset for visualization or further analysis
- Want to avoid multicollinearity in modeling

**Logic:**
- Finds new "axes" (principal components) that explain maximum variance
- First PC explains most variation, second PC explains second-most, etc.
- Components are orthogonal (uncorrelated with each other)

---

### Scree Plot
**Purpose:** Identify how many principal components are needed.

**What to look for:**
- **Elbow point:** Where the slope flattens out (adding more PCs gives diminishing returns)
- **Steep initial slope:** First few PCs capture most variance
- **Flat tail:** Later PCs contribute little

**Interpretation:**
- 1-2 dominant PCs → variation is simple (mainly 1-2 dimensions)
- 5+ PCs needed → variation is complex (many dimensions)

**Implementation example:**
```r
pca_result <- prcomp(scaled_data, scale. = TRUE)
plot(pca_result, type = "l", main = "Scree Plot")
```

---

### Cumulative Variance Explained
**Purpose:** Determine how many PCs to retain based on a threshold (often 80% or 90%).

**Interpretation:**
- If first 2 PCs explain 85% of variance → You can represent data using just 2 dimensions
- If you need 10 PCs for 85% → Data inherently high-dimensional

**Implementation example:**
```r
var_explained <- (pca_result$sdev^2) / sum(pca_result$sdev^2)
cumsum_var <- cumsum(var_explained)

plot(cumsum_var, type = "o", ylim = c(0, 1))
abline(h = 0.8, col = "red")  # 80% threshold
```

---

### Biplot
**Purpose:** Visualize PC loadings (which original variables contribute to each PC) and scores (where each observation falls).

**What to look for:**
- **Arrows pointing same direction:** Variables are correlated
- **Perpendicular arrows:** Variables are uncorrelated
- **Long vs short arrows:** Variables with more/less influence on PCs
- **Clustering in scores:** Groups of similar observations

**Implementation example:**
```r
biplot(pca_result, main = "PCA Biplot")
```

---

### PCA Interpretation Example
**Scenario:** Analyze body measurements (length, width, height, mass, density).

**Expected results:**
- PC1 might capture overall "size" (all body measurements load positively)
- PC2 might capture "shape" (length loads positive, width loads negative)
- **Insight:** Individual variation is mainly in overall size, secondarily in shape

---


## 7. CLUSTERING ANALYSIS

### Purpose of Clustering
**Overall goal:** Group observations based on similarity; identify if natural clusters exist in the data.

**When to use:**
- Want to identify subgroups or patterns
- Dataset contains distinct types/categories not yet identified
- Need to segment population into similar groups

---

### Hierarchical Clustering Dendrogram
**Purpose:** Visualize how observations group together at different similarity levels.

**Logic:**
- Starts with each observation as separate cluster
- Progressively merges most similar clusters
- Creates tree structure showing merge sequence and distances

**What to look for:**
- **Long vertical lines:** Major splits; clusters that are quite different
- **Short vertical lines:** Fine subdivisions; very similar subgroups
- **Natural gap/elbow:** Suggests natural number of clusters to retain

**Distance methods affect results:**
- **Ward.D2:** Minimizes variance within clusters (common choice)
- **Complete:** Maximum distance between clusters
- **Average:** Average distance between clusters

**Implementation example:**
```r
# Scale data
data_scaled <- scale(mydata)

# Calculate distance matrix
dist_matrix <- dist(data_scaled, method = "euclidean")

# Hierarchical clustering
hc <- hclust(dist_matrix, method = "ward.D2")

# Plot dendrogram
plot(hc, main = "Hierarchical Clustering")

# Cut into k clusters
clusters <- cutree(hc, k = 3)  # Extract 3 clusters
```

**Interpretation:** Squids = observations, body measurements = features; height of merge shows dissimilarity.

---

### Cluster Characterization
**Purpose:** Understand what distinguishes the identified clusters.

**What to examine:**
- Size of clusters (balanced or imbalanced?)
- Mean values of key variables for each cluster
- Whether clusters align with external variables (categories, conditions)

**Implementation example:**
```r
# Add cluster assignments
mydata$cluster <- clusters

# Compare mean values across clusters
by(mydata, mydata$cluster, summary)

# Visualization
boxplot(response ~ cluster, data = mydata)
```

**Interpretation:** Helps name or characterize clusters (e.g., "small-immature" vs "large-mature").

---

### Other Clustering Methods
**K-means:** (Alternative to hierarchical)
```r
# K-means clustering (specify k in advance)
kmeans_result <- kmeans(data_scaled, centers = 3)
clusters <- kmeans_result$cluster
```

**Advantages:** Faster; each observation assigned to exactly one cluster.  
**Disadvantage:** Must pre-specify number of clusters.

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
