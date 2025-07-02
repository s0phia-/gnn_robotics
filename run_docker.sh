#!/usr/bin/env bash

FILE_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
OUTPUT_DIR="${FILE_DIR}/results"
SOURCE_DIR="${FILE_DIR}/src"

USE_HARE=0

# if USE_HARE is set to 1, replace all instances of 'docker' with hare
if [ "$USE_HARE" -eq 1 ]; then
    DOCKER_CMD="hare"
else
    DOCKER_CMD="docker"
fi

$DOCKER_CMD run --name isaac-lab -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -v "${SOURCE_DIR}:/workspace/isaaclab/custom_scripts:rw" \
   -v "${OUTPUT_DIR}:/workspace/isaaclab/logs:rw" \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   nvcr.io/nvidia/isaac-lab:2.1.0 \
   bash -c "cd /workspace/isaaclab && ./isaaclab.sh --install"

# change the user 

# Can be used like below - will have to modify our own scripts as everything seems to be relative to require ./isaaclab.sh
# ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64 --headless --video
#
# # in the docker container run the following:
# # - python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Ant-v0 --num_envs 64 --headless --video
# $DOCKER_CMD exec -it isaac-lab bash -c "cd /workspace/isaaclab && ./isaaclab.sh --install"


# run the following inside the docker container for a demo training the ant - should generate video in the 
# python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Ant-v0 --num_envs 64 --headless --video

