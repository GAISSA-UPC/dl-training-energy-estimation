# Install required packages R Markdown notebooks

required_packages <- c(
  "knitr",
  "rmarkdown",
  "languageserver",
  "arrow",
  "tidyverse",
  "lme4",
  "lmerTest",
  "betareg",
  "robustbetareg",
  "gamlss",
  "glmmTMB",
  "DHARMa",
  "effects",
  "broom",
  "car",
  "MASS",
  "MuMIn",
  "performance",
  "emmeans",
  "multcomp",
  "multcompView",
  "h2o",
  "gridExtra",
  "ggeffects",
  "ggplot2",
  "ggstar",
  "ggfortify",
  "ggthemes",
  "patchwork",
  "showtext",
  "Hmisc"
)

installed_packages <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
}

install.packages("internalRmdTools_0.1.0.tar.gz", repos = NULL, type = "source")
