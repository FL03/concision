FROM debian

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y \
    curl \
    git \
    gnupg \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*