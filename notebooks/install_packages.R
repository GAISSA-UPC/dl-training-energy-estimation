# Install required packages R Markdown notebooks

required_packages <- c(
  "arrow",
  "tidyverse",
  "lme4",
  "lmerTest",
  "betareg",
  "robustbetareg",
  "gamlss",
  "glmmTMB",
  "DHARMa",
  "broom",
  "car",
  "MASS",
  "performance",
  "emmeans",
  "multcomp",
  "multcompView",
  "h2o",
  "gridExtra",
  "ggeffects",
  "ggplot2",
  "ggfortify",
  "ggpubr"
)

installed_packages <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
}
