# Cloth Environment

Here is where we will put in the actual environments.

See documentation here: https://gym.openai.com/docs/

TODO:

- Double check representation of the observation space ...
- Actions are cardinal directions for now, in order of N/E/S/W. Is that fine?
- We're only grasping at corners for initial state.


Examples of observation spaces for CartPole and Pong (an Atari game):

```
In [1]: import gym

In [2]: env = gym.make('CartPole-v0')

In [3]: env.observation_space
Out[3]: Box(4,)

In [4]: env = gym.make('PongNoFrameskip-v4')

In [5]: env.observation_space
Out[5]: Box(210, 160, 3)

It seems like we'd rather want something like Box(x,y) rather than Box(x*y)?
```


Relevant links:

- https://github.com/openai/gym/tree/master/gym/spaces
