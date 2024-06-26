from itertools import repeat
from typing import Sequence, Union
import torch
import numpy as np
import torch_geometric.transforms as T


def random_rotate(data, degrees, axis):
    assert data.pos is not None

    degree = np.pi * np.random.uniform(*degrees) / 180.0
    sin, cos = np.sin(degree), np.cos(degree)

    if data.pos.size(-1) == 2:
        matrix = [[cos, sin], [-sin, cos]]
    else:
        if axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

    return T.LinearTransformation(torch.tensor(matrix))(data), matrix


def random_scale(data, scale):
    assert data.pos is not None

    factor = np.random.uniform(*scale)
    data.pos = factor * data.pos

    return data, factor


def random_translate(data, translate):
    assert data.pos is not None
    num_nodes, dim = data.pos.size()

    translate: Sequence[Union[float, int]]
    if isinstance(translate, (int, float)):
        translate = list(repeat(translate, times=dim))
    else:
        assert len(translate) == dim
        translate = translate

    jitter = data.pos.new_empty(num_nodes, dim)
    for d in range(dim):
        jitter[:, d].uniform_(-abs(translate[d]), abs(translate[d]))

    data.pos = data.pos + jitter

    return data, jitter
