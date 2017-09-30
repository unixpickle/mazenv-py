# mazenv-py

A Python port of [unixpickle/mazenv](https://github.com/unixpickle/mazenv) that uses OpenAI Gym.

# Installation

You can install mazenv with pip:

```
pip install mazenv
```

# Usage

You can generate a random 8x8 maze like this:

```
import mazenv

maze = mazenv.prim((8, 8))
print(maze)
```

You can create a Gym environment out of your maze like this:

```
env = mazenv.Env(maze)
```

If you want to restrict the observations to a 5x5 grid centered around the current position, you can do:

```
restricted_env = mazenv.HorizonEnv(env, horizon=2)
```
