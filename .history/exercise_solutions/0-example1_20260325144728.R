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