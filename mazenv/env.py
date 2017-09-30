"""
RL environments for mazes.
"""

import gym
import gym.spaces as spaces
import numpy as np

from .maze import iterate_neighbors

NUM_CELL_FIELDS = 5
ACTION_NOP = 0

class Env(gym.Env):
    """
    Base class for maze environments.

    Observations are the entire maze, represented as a
    Tensor with different entries corresponding to
    different types of objects.

    Actions are discrete with 2*num_dims + 1 options.
    This covers every dimension and a NOP.
    """
    def __init__(self, maze):
        assert maze.start_pos
        obs_shape = maze.shape + (NUM_CELL_FIELDS,)
        self.maze = maze
        self.observation_space = spaces.Box(0, 1, shape=obs_shape)
        self.action_space = spaces.Discrete(len(maze.shape)*2 + 1)
        self.position = maze.start_pos

    def _reset(self):
        self.position = self.maze.start_pos

    def _step(self, action):
        if action != ACTION_NOP:
            new_pos = list(iterate_neighbors(self.position))[action-1]
            if not self.maze.is_wall(new_pos):
                self.position = new_pos
        obs = np.array(self.observation_space.low.shape, dtype='uint8')
        for position in self.maze.positions():
            if self.maze.is_wall(position):
                obs[position][1] = 1
            elif position == self.maze.start_pos:
                obs[position][2] = 1
            elif position == self.maze.end_pos:
                obs[position][3] = 1
            else:
                obs[position][0] = 1
            if position == self.position:
                obs[position][4] = 1
        done = (self.position == self.maze.env_pos)
        rew = -1
        if done:
            rew = 0
        return obs, rew, done, {}
