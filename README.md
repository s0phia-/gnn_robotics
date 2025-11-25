
# Running the Docker Container on Hex
Mostly by James Elson, a bit by Sophia Jones

Note: must be run on a machine with a gpu
- clone and navigate to this repository
- run 'chmod +x run_docker.sh'
- Run the container with `./run_docker.sh`.
- Once it has spun up, you can enter it with `docker exec -it isaac-lab bash` then run scripts
- For example you can run the below command to train on cartpole with video output:

```bash
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video --video_length 100 --video_interval 500
```
