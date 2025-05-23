library(arrow)
library(dplyr)
library(ARTool, pos = 2, lib.loc = NULL)

df <- read_parquet("data/metrics/processed/clean-dl-training-energy-consumption-dataset.gzip") %>%
  filter(architecture != "inception_v3" & `training environment` != "Local Normal User") %>%
  mutate(
    energy=`energy (MJ)`,
    training_environment= factor(`training environment`),
    base_model=factor(architecture),
    subject=factor(case_when(
            architecture == "mobilenet_v2" ~ "model_1",
            architecture == "nasnet_mobile" ~ "model_2",
            architecture == "resnet50" ~ "model_3",
            architecture == "xception" ~ "model_4",
            architecture == "vgg16" ~ "model_5",
          ))
    ) %>%
    group_by(subject, training_environment) %>%
    sample_n(20)

# Count number of observations per subject
df %>%
  group_by(subject) %>%
  summarise(n = n()) %>%
  arrange(desc(n))
# Count number of observations per subject and training environment
df %>%
  group_by(subject, training_environment) %>%
  summarise(n = n()) %>%
  arrange(desc(n))
# Count number of observations per subject and base model
df %>%
  group_by(subject, base_model) %>%
  summarise(n = n()) %>%
  arrange(desc(n))
# Count number of observations per training environment and base model
df %>%
  group_by(training_environment, base_model) %>%
  summarise(n = n()) %>%
  arrange(desc(n))

m <- art(energy ~ training_environment*base_model+Error(base_model), data=df)
summary(m)

anova(m)
