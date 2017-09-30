"""
Tests for env.py
"""

import unittest

from mazenv.env import CURRENT_CELL_FIELD, Env
from mazenv.maze import parse_2d_maze

class EnvTest(unittest.TestCase):
    """
    Tests for raw environments.
    """
    # pylint: disable=R0914
    def test_2d_trajectory(self):
        """
        Test that all the positions, rewards, and done
        signals are correct throughout a trajectory.
        """
        maze = parse_2d_maze('w...w\n' +
                             '..w.w\n' +
                             'Awx..\n' +
                             '..w..')
        env = Env(maze)
        act_nop, act_up, act_down, act_left, act_right = [0, 1, 2, 3, 4]
        actions = [act_down, act_right, act_right, act_nop, act_left,
                   act_left, act_up, act_up, act_up, act_right, act_up,
                   act_right, act_right, act_down, act_left, act_down,
                   act_nop, act_left]
        positions = [(3, 0), (3, 1), (3, 1), (3, 1), (3, 0),
                     (3, 0), (2, 0), (1, 0), (1, 0), (1, 1), (0, 1),
                     (0, 2), (0, 3), (1, 3), (1, 3), (2, 3),
                     (2, 3), (2, 2)]
        dones = [False]*(len(positions)-1) + [True]
        for act, expected_pos, expected_done in zip(actions, positions, dones):
            obs, rew, done, _ = env.step(act)
            self.assertEqual(done, expected_done)
            if done:
                self.assertEqual(rew, 0.0)
            else:
                self.assertEqual(rew, -1.0)
            self._assert_unique_cell(maze, obs, expected_pos, CURRENT_CELL_FIELD)

    def _assert_unique_cell(self, maze, obs, pos, cell_idx):
        """
        Assert that the cell is set at the position in the
        observation, and nowhere else in the observation.
        """
        for some_pos in maze.positions():
            cell_val = obs[some_pos][cell_idx]
            if some_pos == pos:
                self.assertEqual(cell_val, 1)
            else:
                self.assertEqual(cell_val, 0)

if __name__ == '__main__':
    unittest.main()
