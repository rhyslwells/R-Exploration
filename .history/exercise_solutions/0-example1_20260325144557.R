# Load squid data and perform basic analysis

# Read the squid1.txt file from the data directory into a data frame
# header = TRUE indicates the first row contains column names
# dec = "." specifies the decimal separator (period for numbers like 87.5)
squid <- read.table("data/squid1.txt", header = TRUE)

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

# Create a metadata table showing information about each field
# Initialize vectors to store metadata information
field_names <- names(squid)
data_types <- sapply(squid, class)
missing_values <- colSums(is.na(squid))
non_null_count <- nrow(squid) - missing_values

# Create a data frame to store metadata
metadata <- data.frame(
  Field = field_names,
  Data_Type = data_types,
  Non_Null = non_null_count,
  Missing = missing_values,
  stringsAsFactors = FALSE
)

# Add additional statistics for numeric columns (min, max, mean)
metadata$Min <- NA
metadata$Max <- NA
metadata$Mean <- NA

for (i in 1:nrow(metadata)) {
  if (metadata$Data_Type[i] == "numeric" || metadata$Data_Type[i] == "integer") {
    metadata$Min[i] <- min(squid[[i]], na.rm = TRUE)
    metadata$Max[i] <- max(squid[[i]], na.rm = TRUE)
    metadata$Mean[i] <- round(mean(squid[[i]], na.rm = TRUE), 2)
  }
}

# Display the metadata table
print(metadata)

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