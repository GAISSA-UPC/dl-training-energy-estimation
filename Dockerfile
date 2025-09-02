# =====================
# Stage 1: Build
# =====================
FROM python:3.10.18-trixie AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential gfortran liblapack-dev libblas-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev zlib1g-dev \
    libpng-dev libjpeg-dev libfreetype6-dev libfontconfig1-dev \
    libharfbuzz-dev libfribidi-dev libtiff5-dev libicu-dev \
    cmake git r-base \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY internalRmdTools_0.1.0.tar.gz .
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt
RUN R -e "pkgs <- c( \
          'knitr','rmarkdown','languageserver','arrow','tidyverse', \
          'lme4','lmerTest','betareg','robustbetareg','gamlss','glmmTMB','DHARMa','effects','effectsize','stringi', \
          'car','MASS','MuMIn','performance','emmeans','multcomp','multcompView','h2o','gridExtra','ggeffects', \
          'ggplot2','ggstar','ggfortify','ggpubr','ggthemes','patchwork','showtext','Hmisc'); \
          install.packages(pkgs, repos='https://cloud.r-project.org'); \
          missing <- pkgs[!sapply(pkgs, requireNamespace, quietly=TRUE)]; \
          if (length(missing) > 0) { \
            message('Missing packages: ', paste(missing, collapse=', ')); \
            quit(status=1) }"

RUN R -e "install.packages('internalRmdTools_0.1.0.tar.gz', repos = NULL, type = 'source')"

# =====================
# Stage 2: Deploy
# =====================
FROM python:3.10.18-trixie

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    r-base \
    liblapack-dev libblas-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev zlib1g-dev \
    libpng-dev libjpeg-dev libfreetype6-dev libfontconfig1-dev \
    libharfbuzz-dev libfribidi-dev libtiff5-dev libicu-dev \
    pandoc \
    fonts-cmu \
    && rm -rf /var/lib/apt/lists/*


# Install R dependencies
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library

# Install Python dependencies
COPY --from=builder /wheels /wheels
RUN pip install --upgrade pip && pip install --no-cache /wheels/*


ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME

WORKDIR /home/$USERNAME/workspace

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV PYTHONPATH="/home/$USERNAME/workspace"

RUN echo "ROOT = /home/$USERNAME/workspace/dl-training-energy-estimation" >> /home/$USERNAME/workspace/.env

COPY --chown=${USERNAME}:${USERNAME} . .
