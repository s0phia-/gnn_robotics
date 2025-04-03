#IGNORE THIS FILE FOR NOW

FROM tensorflow/tensorflow:latest

# Command line apps required
RUN apt update && apt install -y wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev graphviz graphviz-dev patchelf

# Install dependencies for macOS
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create .mujoco directory
RUN mkdir -p /root/.mujoco

# Download and extract MuJoCo 2.1.0 for macOS
RUN wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz \
    && tar -xf mujoco210-macos-x86_64.tar.gz -C /root/.mujoco \
    && rm mujoco210-macos-x86_64.tar.gz

# Set environment variables
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV MUJOCO_PY_MUJOCO_PATH /root/.mujoco/mujoco210

# Install rest of dependencies
RUN pip install pip --upgrade
RUN pip install mujoco-py
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install gym

WORKDIR /home/gnn_robotics