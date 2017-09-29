"""
Tests for maze.py
"""

import unittest
import numpy as np

from .maze import Maze, parse_2d_maze

class TestMazeSerialization(unittest.TestCase):
    """
    Test cases for conversions between mazes and strings.
    """
    def test_str_2d(self):
        """
        Test 2-D maze serialization.
        """
        for maze, expected in _testing_maze_pairs():
            self.assertEqual(str(maze), expected)

    def test_parse(self):
        """
        Test 2-D maze parsing.
        """
        for maze, in_str in _testing_maze_pairs():
            parsed = parse_2d_maze(in_str)
            self.assertEqual(parsed.start_pos, maze.start_pos)
            self.assertEqual(parsed.end_pos, maze.end_pos)
            self.assertTrue((parsed.walls == maze.walls).all())

def _testing_maze_pairs():
    """
    Return tuples of (maze, str).
    """
    res = []
    maze = Maze(np.array([[False, True, False], [True, False, False]]),
                start_pos=(0, 2))
    res.append((maze, '.wA\nw..'))
    maze = Maze(np.array([[False, True], [True, False], [False, True]]),
                start_pos=(1, 1), end_pos=(2, 0))
    res.append((maze, '.w\nwA\nxw'))
    maze = Maze(np.array([[False, True], [True, False], [False, True]]),
                end_pos=(2, 0))
    res.append((maze, '.w\nw.\nxw'))
    return res
