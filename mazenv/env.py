"""
RL environments for mazes.
"""

import itertools

import gym
import gym.spaces as spaces
import numpy as np

from . import _util as util

NUM_CELL_FIELDS = 5
SPACE_CELL_FIELD = 0
WALL_CELL_FIELD = 1
START_CELL_FIELD = 2
END_CELL_FIELD = 3
CURRENT_CELL_FIELD = 4
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
        return self._make_observation()

    def _step(self, action):
        if action != ACTION_NOP:
            new_pos = list(util.iterate_neighbors(self.position))[action-1]
            if not self.maze.is_wall(new_pos):
                self.position = new_pos
        done = (self.position == self.maze.end_pos)
        rew = -1.0
        if done:
            rew = 0.0
        return self._make_observation(), rew, done, {}

    def _make_observation(self):
        """
        Create an observation for the current state.
        """
        obs = np.zeros(self.observation_space.low.shape, dtype='uint8')
        for position in self.maze.positions():
            self._fill_cell(obs[position], position)
        return obs

    def _fill_cell(self, cell, cell_position):
        """
        Set the entries of the observation cell.
        """
        if self.maze.is_wall(cell_position):
            cell[WALL_CELL_FIELD] = 1
        elif cell_position == self.maze.start_pos:
            cell[START_CELL_FIELD] = 1
        elif cell_position == self.maze.end_pos:
            cell[END_CELL_FIELD] = 1
        else:
            cell[SPACE_CELL_FIELD] = 1
        if cell_position == self.position:
            cell[CURRENT_CELL_FIELD] = 1


class HorizonEnv(Env):
    """
    A maze environment that restricts observations to a
    fixed horizon from the current position.

    The number of cells away you can see along an axis is
    called the "horizon".
    For a horizon of 1, observations have side length 3.
    """

    def __init__(self, maze, horizon=1):
        super(HorizonEnv, self).__init__(maze)
        self.horizon = horizon
        self.old_shape = self.observation_space.low.shape[:-1]
        num_dims = len(self.old_shape)
        obs_size = (horizon*2 + 1,) * num_dims + (NUM_CELL_FIELDS,)
        self.observation_space = spaces.Box(0, 1, shape=obs_size)

    def _make_observation(self):
        """
        Compute the indices of each cell in the visible
        horizon grid.

        The cells are generated in order such that they
        can be reshaped to the N-d observation.
        """
        pos = self.position
        ranges = [list(range(x-self.horizon, x+self.horizon+1)) for x in pos]
        grid = np.zeros(self.observation_space.low.shape, dtype='uint8')
        flat_grid = grid.reshape((-1, NUM_CELL_FIELDS))
        for idx, position in enumerate(itertools.product(*ranges)):
            self._fill_cell(flat_grid[idx], position)
        return grid
