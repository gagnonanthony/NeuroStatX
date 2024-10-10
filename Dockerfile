# Building the image from the base image.
FROM ubuntu:22.04

# Defining args.
ARG VERSION

ARG VERSION=${VERSION:-"main"}
ENV REPO="https://github.com/gagnonanthony/NeuroStatX.git"

WORKDIR /
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    apt-get update && DEBIAN_FRONTED=noninteractive apt-get install -y \
        git \
        python3 \
        pip \
        graphviz && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
ADD ${REPO}#${VERSION} /NeuroStatX

WORKDIR /NeuroStatX
RUN pip install .
