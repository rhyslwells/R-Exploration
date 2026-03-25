# Load squid data and perform basic analysis

# Read the squid1.txt file from the data directory into a data frame
# header = TRUE indicates the first row contains column names
# dec = "." specifies the decimal separator (period for numbers like 87.5)
squid <- read.table("data/squid1.txt", header = TRUE)

# **field:: sample.no** — Identifier for the sample collection batch (e.g., 105128901, 116019001). Represents which research collection the specimens came from.

# **field:: specimen** — Unique identifier number for each individual squid specimen measured (e.g., 1002, 1003, 1005).

# **field:: year** — Year the specimen was collected (ranges from 1989-1990 in the data).

# **field:: month** — Month of collection (1-12), ranging from January (1) to December (12).

# **field:: weight** — Total body weight of the squid in grams, ranging from ~54g to over 700g in the dataset.

# **field:: sex** — Coded sex indicator (appears to all be coded as 2, likely indicating female).

# **field:: maturity.stage** — Reproductive maturity classification on a numerical scale (1-5), where higher numbers represent more advanced reproductive development.

# **field:: DML** — Dorsal Mantle Length in millimeters, the standard measurement of squid body length from the top of the mantle.

# **field:: eviscerate.weight** — Weight of the squid after removing internal organs (in grams).

# **field:: dig.weight** — Weight of the digestive system contents or components (in grams).

# **field:: nid.length** — Length of the nidamental gland, a reproductive organ in female squid (in mm).

# **field:: nid.weight** — Weight of the nidamental gland (in grams).

# **field:: ovary.weight** — Weight of the ovary, indicating reproductive investment and development stage (in grams).

# Display the first 6 rows to see the structure of the data
head(squid)

# Display the last 6 rows to see the end of the data
tail(squid)

# Check the dimensions: number of rows (specimens) and columns (variables)
dim(squid)

# Display the column names in the dataset
names(squid)

# Get a summary of each column with basic statistics (mean, median, quartiles, etc.)
summary(squid)

# Check the data types of each column
str(squid)

# Get the range of dates when the specimens were collected
range(squid$year)

# Get the number of rows (number of squid specimens)
nrow(squid)

# Get the number of columns (number of variables measured)
ncol(squid)

# Examine the structure and data types more clearly
class(squid)

# Calculate the mean weight of squids
mean(squid$weight)

# Calculate the standard deviation of weight values
sd(squid$weight)

# Find the minimum weight value in the dataset
min(squid$weight)

# Create a frequency table to see how many specimens by sex (2 = female, presumably)
table(squid$sex)

# Create a frequency table of maturity stages
table(squid$maturity.stage)

# Create a box plot of weight by maturity stage to visualize the relationship
boxplot(squid$weight ~ squid$maturity.stage, 
        main = "Squid Weight by Maturity Stage",
        xlab = "Maturity Stage", 
        ylab = "Weight")

# Create a scatter plot of DML (body length) vs weight
plot(squid$DML, squid$weight,
     main = "Squid DML vs Weight",
     xlab = "Dorsal Mantle Length (DML)",
     ylab = "Weight",
     pch = 16)

# Add a linear regression line to the scatter plot
abline(lm(squid$weight ~ squid$DML), col = "red")

# Create a histogram of DML values
hist(squid$DML,
     main = "Histogram of DML Values",
     xlab = "Dorsal Mantle Length (DML)",
     ylab = "Frequency",
     breaks = 20)

# ============================================================================
# CORRELATION & REGRESSION ANALYSIS
# ============================================================================

# Create a correlation matrix for numeric variables
# cor() calculates Pearson correlation coefficients between all numeric variables
numeric_cols <- squid[, c("weight", "DML", "eviscerate.weight", "dig.weight", 
                          "nid.length", "nid.weight", "ovary.weight")]
correlation_matrix <- cor(numeric_cols, use = "complete.obs")
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix as a heatmap
heatmap(correlation_matrix, main = "Correlation Heatmap of Squid Measurements")

# Simple linear regression: weight as a function of DML
lm_model <- lm(weight ~ DML, data = squid)
summary(lm_model)

# Multiple linear regression: weight predicted by DML, ovary.weight, and maturity.stage
lm_multiple <- lm(weight ~ DML + ovary.weight + as.factor(maturity.stage), data = squid)
summary(lm_multiple)

# ============================================================================
# GROUP COMPARISONS (ANOVA)
# ============================================================================

# One-way ANOVA: Test if weight differs significantly across maturity stages
# H0: mean weight is the same for all maturity stages
anova_weight <- aov(weight ~ as.factor(maturity.stage), data = squid)
summary(anova_weight)

# One-way ANOVA: Test if DML differs significantly across maturity stages
anova_dml <- aov(DML ~ as.factor(maturity.stage), data = squid)
summary(anova_dml)

# Post-hoc Tukey HSD test to identify which maturity stages differ
tukey_result <- TukeyHSD(anova_weight)
print(tukey_result)

# Pairwise t-tests with Bonferroni correction
pairwise_t_tests <- pairwise.t.test(squid$weight, squid$maturity.stage, 
                                     p.adjust.method = "bonferroni")
print(pairwise_t_tests)

# ============================================================================
# DISTRIBUTIONS & ASSUMPTIONS TESTING
# ============================================================================

# Shapiro-Wilk test for normality of weight distribution
# H0: data is normally distributed
shapiro_weight <- shapiro.test(squid$weight)
print("Shapiro-Wilk test for weight normality:")
print(shapiro_weight)

# Shapiro-Wilk test for normality of DML distribution
shapiro_dml <- shapiro.test(squid$DML)
print("Shapiro-Wilk test for DML normality:")
print(shapiro_dml)

# Create density plot for weight with normal curve overlay
plot(density(squid$weight), main = "Density Plot of Weight")
# Add a theoretical normal distribution curve
curve(dnorm(x, mean = mean(squid$weight), sd = sd(squid$weight)), 
      col = "red", add = TRUE, lty = 2)

# Create density plot for DML with normal curve overlay
plot(density(squid$DML), main = "Density Plot of DML")
curve(dnorm(x, mean = mean(squid$DML), sd = sd(squid$DML)), 
      col = "red", add = TRUE, lty = 2)

# Q-Q plots to visualize departures from normality
qqnorm(squid$weight, main = "Q-Q Plot of Weight")
qqline(squid$weight, col = "red")

qqnorm(squid$DML, main = "Q-Q Plot of DML")
qqline(squid$DML, col = "red")

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================

# Compare measurements between years (1989 vs 1990)
# Create a box plot of weight by year
boxplot(squid$weight ~ squid$year, 
        main = "Squid Weight by Year",
        xlab = "Year", 
        ylab = "Weight")

# T-test comparing weight between years
weight_year_test <- t.test(squid$weight[squid$year == 1989], 
                            squid$weight[squid$year == 1990])
print("T-test comparing weight between years:")
print(weight_year_test)

# Compare measurements by month (seasonal patterns)
boxplot(squid$weight ~ squid$month, 
        main = "Squid Weight by Month",
        xlab = "Month", 
        ylab = "Weight")

# ANOVA to test if weight differs by month
anova_month <- aov(weight ~ as.factor(month), data = squid)
summary(anova_month)

# ============================================================================
# REPRODUCTIVE INVESTMENT ANALYSIS
# ============================================================================

# Calculate allometric relationship: ovary weight vs body size
plot(squid$DML, squid$ovary.weight,
     main = "Reproductive Allometry: Ovary Weight vs DML",
     xlab = "Dorsal Mantle Length (DML)",
     ylab = "Ovary Weight (g)",
     pch = 16)
# Add regression line
abline(lm(squid$ovary.weight ~ squid$DML), col = "blue", lwd = 2)

# Linear regression for allometric relationship
allometry_model <- lm(ovary.weight ~ DML, data = squid)
summary(allometry_model)

# Box plot: Ovary weight by maturity stage
boxplot(squid$ovary.weight ~ squid$maturity.stage,
        main = "Ovary Weight by Maturity Stage",
        xlab = "Maturity Stage",
        ylab = "Ovary Weight (g)")

# Calculate reproductive investment ratio (ovary weight / body weight)
squid$repro_investment <- squid$ovary.weight / squid$weight

# Box plot of reproductive investment ratio by maturity stage
boxplot(squid$repro_investment ~ squid$maturity.stage,
        main = "Reproductive Investment Ratio by Maturity Stage",
        xlab = "Maturity Stage",
        ylab = "Ovary Weight / Body Weight")

# ANOVA on reproductive investment across maturity stages
anova_repro <- aov(repro_investment ~ as.factor(maturity.stage), data = squid)
summary(anova_repro)

# Correlation between nidamental gland weight and ovary weight
plot(squid$nid.weight, squid$ovary.weight,
     main = "Correlation: Nidal Weight vs Ovary Weight",
     xlab = "Nidamental Gland Weight (g)",
     ylab = "Ovary Weight (g)",
     pch = 16)
abline(lm(squid$ovary.weight ~ squid$nid.weight), col = "green", lwd = 2)

# Calculate correlation
nid_ovary_cor <- cor(squid$nid.weight, squid$ovary.weight)
print(paste("Correlation between nid weight and ovary weight:", round(nid_ovary_cor, 3)))

# ============================================================================
# MULTIVARIATE ANALYSIS - PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================

# Prepare data for PCA (standardize numeric columns)
pca_data <- squid[, c("weight", "DML", "eviscerate.weight", "dig.weight", 
                      "nid.length", "nid.weight", "ovary.weight")]

# Remove any rows with missing values
pca_data_clean <- na.omit(pca_data)

# Perform PCA on standardized data
pca_result <- prcomp(pca_data_clean, scale. = TRUE)

# Display PCA summary
summary(pca_result)

# Scree plot: shows variance explained by each PC
plot(pca_result, type = "l", main = "PCA Scree Plot")

# Biplot: shows loadings and scores
biplot(pca_result, main = "PCA Biplot")

# Variance explained by each PC
var_explained <- (pca_result$sdev^2) / sum(pca_result$sdev^2)
print("Proportion of Variance Explained by Each PC:")
print(var_explained)

# Cumulative variance explained
cumsum_var <- cumsum(var_explained)
plot(cumsum_var, ylim = c(0, 1), type = "o", 
     main = "Cumulative Variance Explained by PCA",
     xlab = "Principal Component",
     ylab = "Cumulative Variance Explained")
abline(h = 0.8, col = "red", lty = 2)
abline(h = 0.9, col = "red", lty = 2)

# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

# Hierarchical clustering based on standardized measurements
# Scale the data
squid_scaled <- scale(pca_data_clean)

# Calculate distance matrix
dist_matrix <- dist(squid_scaled, method = "euclidean")

# Perform hierarchical clustering
hc_result <- hclust(dist_matrix, method = "ward.D2")

# Plot dendrogram
plot(hc_result, main = "Hierarchical Clustering Dendrogram",
     xlab = "Specimen", ylab = "Distance")

# Cut dendrogram to create 3 clusters
clusters <- cutree(hc_result, k = 3)

# Add cluster assignments to data
squid$cluster <- NA
squid$cluster[as.numeric(rownames(pca_data_clean))] <- clusters

# Box plot: weight by cluster
boxplot(squid$weight ~ squid$cluster,
        main = "Weight Distribution by Cluster",
        xlab = "Cluster",
        ylab = "Weight")

# ============================================================================
# ADDITIONAL EXPLORATORY VISUALIZATIONS
# ============================================================================

# Violin plot: weight by maturity stage (better than box plot for distributions)
library(graphics)

# Create a matrix of scatter plots (pairs plot)
pairs(squid[, c("weight", "DML", "ovary.weight", "nid.weight")],
      main = "Pairwise Scatter Plots of Key Variables",
      pch = 16)

# Grouped scatter plot: DML vs weight, colored by maturity stage
plot(squid$DML, squid$weight,
     col = squid$maturity.stage,
     main = "DML vs Weight by Maturity Stage",
     xlab = "DML (mm)",
     ylab = "Weight (g)",
     pch = 16)
legend("topleft", legend = unique(squid$maturity.stage),
       col = unique(squid$maturity.stage), pch = 16,
       title = "Maturity Stage")

# Summary statistics by maturity stage
print("Summary of weight by maturity stage:")
by(squid$weight, squid$maturity.stage, summary)

print("Summary of ovary weight by maturity stage:")
by(squid$ovary.weight, squid$maturity.stage, summary)

# Create contingency table: year vs maturity stage
contingency_table <- table(squid$year, squid$maturity.stage)
print("Contingency Table: Year vs Maturity Stage")
print(contingency_table)

# Chi-square test of independence
chi_sq <- chisq.test(contingency_table)
print("Chi-square test of independence:")
print(chi_sq)

# Calculate the correlation between DML and weight
cor(squid$DML, squid$weight)

#------------------------------------------------

