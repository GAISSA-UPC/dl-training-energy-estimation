FROM python:3.10.18-trixie AS python-builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    libssl-dev

COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

FROM rocker/tidyverse:4.5.1 AS r-builder

RUN install2.r --error \
    knitr \
    rmarkdown \
    languageserver \
    arrow \
    tidyverse \
    lme4 \
    lmerTest \
    betareg \
    robustbetareg \
    gamlss \
    glmmTMB \
    DHARMa \
    effects \
    broom \
    car \
    MASS \
    MuMIn \
    performance \
    emmeans \
    multcomp \
    multcompView \
    h2o \
    gridExtra \
    ggeffects \
    ggplot2 \
    ggstar \
    ggfortify \
    ggthemes \
    patchwork \
    showtext \
    Hmisc

COPY internalRmdTools_0.1.0.tar.gz ./
RUN R -e "install.packages('internalRmdTools_0.1.0.tar.gz', repos = NULL, type = 'source')"

FROM python:3.10.18-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    r-base \
    fonts-cmu \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY --from=python-builder /wheels /wheels
RUN pip install --upgrade pip && pip install --no-cache /wheels/*

# Install R dependencies
COPY --from=r-builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library

WORKDIR /workspace

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --ingroup appgroup appuser && \
    chown -R appuser:appgroup /workspace

USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"
