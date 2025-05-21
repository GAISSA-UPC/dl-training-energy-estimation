library(dplyr)
library(arrow)
library(ARTool)
library(car)

df <- read_parquet("clean-dl-training-energy-consumption-dataset.gzip") %>%
  filter(architecture != "inception_v3" & `training environment` != "Local Normal User") %>%
  mutate(energy=`energy (MJ)`, training_environment= factor(`training environment`), architecture=factor(architecture))

head(df)

m <- art(energy ~ training_environment*architecture, data=df)

summary(m)

anova(m)

art.con(m, "training_environment")
art.con(m, "training_environment:architecture") %>%
  summary() %>%
  mutate(sig = symnum(p.value, corr = FALSE, na = FALSE,
                      cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                      symbols = c("***", "**", "*", ".", " ")
  ))
