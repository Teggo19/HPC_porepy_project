FROM ubuntu:24.04

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/HPC_porepy_project

# System dependencies and legacy FEniCS from archived installation route
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        git \
        wget \
        pkg-config \
        g++ \
        libglu1-mesa \
        libgl1 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libx11-6 \
        libxcursor1 \
        libxinerama1 \
        libxi6 \
        libxrandr2 \
        libxft2 \
        libfontconfig1 \
        libfreetype6 \
        libxfixes3 \
        python3-venv \
        python3-pip \
        python3-dev \
    && add-apt-repository -y ppa:fenics-packages/fenics \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        fenics \
    && wget -q https://github.com/precice/precice/releases/download/v3.4.0/libprecice3_3.4.0_noble.deb \
    && apt-get install -y ./libprecice3_3.4.0_noble.deb \
    && rm -f libprecice3_3.4.0_noble.deb \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /workspace/HPC_porepy_project

# FreeFlow environment (reuses system DOLFIN from base image)
RUN python3 -m venv /opt/venvs/freeflow --system-site-packages \
    && source /opt/venvs/freeflow/bin/activate \
    && pip install --upgrade pip \
    && pip install fenicsprecice pyprecice

# PorousMedia environment
RUN python3 -m venv /opt/venvs/porous \
    && source /opt/venvs/porous/bin/activate \
    #&& pip install --upgrade pip \
    #&& pip install --upgrade setuptools wheel build \
    && pip install numpy matplotlib pyprecice \
    && git clone --depth 1 https://github.com/pmgbergen/porepy.git /tmp/porepy-src \
    && cd /tmp/porepy-src \
    && pip install . 
    #&& python -c "import porepy; print('porepy source install ok')" \
    #&& rm -rf /tmp/porepy-src

RUN chmod +x /workspace/HPC_porepy_project/docker/run_coupled.sh

CMD ["/bin/bash", "-lc", "/workspace/HPC_porepy_project/docker/run_coupled.sh"]
