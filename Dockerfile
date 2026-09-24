# Building the image from the base image.
FROM python:3.12-slim

# Defining args.
ARG VERSION

ARG VERSION=${VERSION:-"main"}
ENV REPO="https://github.com/gagnonanthony/NeuroStatX.git"

WORKDIR /
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    apt-get update && DEBIAN_FRONTED=noninteractive apt-get install -y \
        git \
        graphviz && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN git clone --branch ${VERSION} ${REPO}

WORKDIR /NeuroStatX
RUN pip install .
