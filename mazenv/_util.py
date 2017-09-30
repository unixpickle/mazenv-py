"""
Utilities for grids.
"""

def iterate_positions(shape):
    """
    Generate an iterable of all valid indices in a tensor.
    """
    # TODO: use itertools.product() here.
    for i in range(shape[0]):
        if len(shape) == 1:
            yield (i,)
        else:
            for sub_idx in iterate_positions(shape[1:]):
                yield (i,) + sub_idx

def iterate_neighbors(pos):
    """
    Iterate through all the single-axis neighbors of a
    coordinate tuple.

    Order is deterministic.
    """
    for axis, val in enumerate(pos):
        for offset in [-1, 1]:
            yield pos[:axis] + (val+offset,) + pos[axis+1:]

def shape_contains(shape, pos):
    """
    Check if a tensor of the given shape contains the
    position.
    """
    assert len(shape) == len(pos)
    for val, max_val in zip(pos, shape):
        if val < 0 or val >= max_val:
            return False
    return True
