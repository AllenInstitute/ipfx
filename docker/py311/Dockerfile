FROM python:3.11

RUN apt-get update \
    && apt-get install -y \
        hdf5-tools \
        curl

RUN curl https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN apt-get install git-lfs
