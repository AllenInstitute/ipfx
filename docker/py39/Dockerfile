FROM python:3.9

RUN apt-get update \
    && apt-get install -y \
        hdf5-tools \
        curl \
        git-lfs \
        && rm -rf /var/lib/apt/lists/*

RUN curl https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash