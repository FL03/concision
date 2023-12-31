FROM debian

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y \
    build-essential \
    curl \
    git \
    gnupg \
    libprotobuf-dev \
    libssl-dev \
    pkg-config \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y \
    liblapack-dev \
    libopenblas-dev \
