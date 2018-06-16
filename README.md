# tf-env

RL environments implemented in pure TensorFlow graphs.

# Pong environment

The pong environment is a proof of concept re-implementation of the Pong video game in TensorFlow.

![Pong GIF](example/pong_gameplay.gif)

## Trainability

The script [example/pong_ppo.py](example/pong_ppo.py) trains an agent on the Pong environment. On a decent GPU, it should take less than 10 minutes to get good at the game.

## Performance

The script [example/pong_bench.py](example/pong_bench.py) measures the FPS of the Pong environment.

Here are the benchmark results on a Tesla K80 GPU:

```
fps is 6087.651217 with render=False
fps is 3734.036180 with render=True
```

Here are the benchmark results on an old, 2.6GHz Core i7:

```
fps is 22324.623265 with render=False
fps is 956.894775 with render=True
```
