
FROM rust:latest AS builder-base

RUN apt-get update -y && \
    apt-get upgrade -y

RUN apt-get install -y \
    clang \
    cmake \
    git \
    libclang-dev \
    libopenblas-dev \
    libssl-dev \
    pkg-config

RUN rustup update

FROM builder-base AS builder

ADD . /app
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build -r --features full --workspace

