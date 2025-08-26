ARG PYTHON_VERSION=XXX
FROM python:${PYTHON_VERSION}

ARG APP_UID=1001

ARG APP_GID=1001

RUN apt-get update                    \
    && apt-get install -y             \
        hdf5-tools                    \
        curl                          \
        git-lfs                       \
        && rm -rf /var/lib/apt/lists/*

RUN groupadd -g "$APP_GID" ci                               && \
    useradd -l -u "$APP_UID" -g "$APP_GID" -ms /bin/bash ci

RUN curl https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

USER ci

WORKDIR /home/ci
