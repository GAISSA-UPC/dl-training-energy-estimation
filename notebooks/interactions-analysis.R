library(arrow)
library(dplyr)
library(lme4)
library(lmerTest)      # p-values for lmer
library(performance)   # check assumptions
library(emmeans)       # post hoc tests
library(car)
library(ARTool)
library(ggplot2)

df <- read_parquet("data/metrics/processed/clean-dl-training-energy-consumption-dataset.gzip") %>%
  filter(architecture != "inception_v3" & `training environment` != "Local Normal User") %>%
  mutate(
    energy=`energy (MJ)`,
    training_environment= factor(`training environment`),
    base_model=factor(architecture),
    ) %>%
    group_by(base_model, training_environment) %>%
    mutate(
      subject_id = factor(row_number()),
    )


model_lmm <- lmer(energy ~ base_model * training_environment + (1 | subject_id), data = df)
summary(model_lmm)

# QQ plot
qqnorm(resid(model_lmm)); qqline(resid(model_lmm))

# Residuals histogram
hist(resid(model_lmm), main = "Residuals Histogram", xlab = "Residuals")

# Residuals vs Fitted
plot(fitted(model_lmm), resid(model_lmm), main = "Residuals vs Fitted", xlab = "Fitted", ylab = "Residuals")
abline(h = 0, col = "red")

# Shapiro-Wilk test
normality_results <- shapiro.test(resid(model_lmm))  # If p < 0.05, consider transformation

# Comprehensive model checks
check_model(model_lmm)

if (normality_results$p.value < 0.05) {
  cat("Residuals are not normally distributed. Consider transformation.\n")
  df <- df %>% mutate(log_energy = log(energy + 1e-6))
  model_lmm_log <- lmer(log_energy ~ base_model * training_environment + (1 | subject_id), data = df)
  summary(model_lmm_log)
  check_model(model_lmm_log)
}

emmeans(model_lmm, pairwise ~ base_model | training_environment)
emmeans(model_lmm, pairwise ~ training_environment | base_model)

# # Count number of observations per training environment and base model
# df %>%
#   group_by(training_environment, base_model) %>%
#   summarise(n = n()) %>%
#   arrange(desc(n))

# m <- art(energy ~ training_environment*base_model, data=df)
# summary(m)

# anova(m)
