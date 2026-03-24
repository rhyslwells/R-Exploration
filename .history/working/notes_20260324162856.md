How to generate Rmd files from R scripts

how to generate a quick github pahes from html files from Rmd files.

Use https://alexd106.github.io/intro2R/index.html as an example.

.qmd files?

## Using R

plots

## stats

test for normal distribution:
shapiro.test()

Compare two variances: var.test() F test

linear models of statisical tests


response variable ~ explanatory variable(s) + error
• literally read as ‘variation in response variable modelled as a 
function of the explanatory variable(s) plus variation not explained
by the explanatory variable

the response variable comes first, then the tilde ~ then the name
of the explanatory variable
clouds.lm <- lm(moisture ~ treatment, data=clouds

 normality of residuals
- use histograms and Q-Q plots of the residual