#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-23 19:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
import torch as th


def sl(dims, axis, idx=None):
    """slice any axis

    generates slice of specified axis.

    Parameters
    ----------
    dims : int
        total dimensions
    axis : {int or list}
        select axis list.
    idx : list or None, optional
        slice lists of the specified :attr:`axis`, if None, does nothing (the default)

    Returns
    -------
    [tuple of slice]
        slice for specified axis elements.
    """

    idxall = [slice(None)] * dims

    axis = [axis] if type(axis) is int else axis
    idx = [idx] if type(idx) not in [list, tuple] else idx
    if len(axis) != len(idx):
        raise ValueError('The index for each axis should be given!')

    naxis = len(axis)
    for n in range(naxis):
        idxall[axis[n]] = idx[n]

    return tuple(idxall)


def cut(x, pos, axis=None):
    """Cut array at given position.

    Cut array at given position.

    Parameters
    ----------
    x : {torch.tensor}
        [description]
    pos : {tulpe or list}
        cut positions: ((cpstart, cpend), (cpstart, cpend), ...)
    axis : {number, tulpe or list}, optional
        cut axis (the default is None, which means nothing)
    """

    if axis is None:
        return x
    if type(axis) == int:
        axis = tuple([axis])
    nDims = x.dim()
    idx = [None] * nDims

    if len(axis) > 1 and len(pos) != len(axis):
        raise ValueError('You should specify cut axis for each cut axis!')
    elif len(axis) == 1:
        axis = tuple(list(axis) * len(pos))

    uqaixs = np.unique(axis)
    for a in uqaixs:
        idx[a] = []

    for i in range(len(axis)):
        idx[axis[i]] += range(pos[i][0], pos[i][1])

    for a in uqaixs:
        idxall = [slice(None)] * nDims
        idxall[a] = idx[a]
        x = x[tuple(idxall)]
    return x


def arraycomb(arrays, out=None):
    arrays = [x if type(x) is th.Tensor else th.tensor(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.numel() for x in arrays])
    if out is None:
        out = th.zeros([n, len(arrays)], dtype=dtype)
    m = int(n / arrays[0].numel())
    out[:, 0] = arrays[0].repeat_interleave(m)

    if arrays[1:]:
        arraycomb(arrays[1:], out=out[0:m, 1:])

    for j in range(1, arrays[0].numel()):
        out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]

    return out


if __name__ == '__main__':

    X = th.randint(0, 100, (9, 10))
    print('X')
    print(X)
    Y = cut(X, ((1, 4), (5, 8)), axis=0)
    print('Y = cut(X, ((1, 4), (5, 8)), axis=0)')
    print(Y)
    Y = cut(X, ((1, 4), (7, 9)), axis=(0, 1))
    print('Y = cut(X, ((1, 4), (7, 9)), axis=(0, 1))')
    print(Y)
    Y = cut(X, ((1, 4), (1, 4), (5, 8), (7, 9)), axis=(0, 1, 0, 1))
    print('cut(X, ((1, 4), (1, 4), (5, 8), (7, 9)), axis=(0, 1, 0, 1))')
    print(Y)

    print(X[sl(2, -1, [0, 1])])
    print(X[:, 0:2])

    x = arraycomb(([1, 2, 3, 4], [4, 5], [6, 7]))
    print(x, x.shape)

    x = arraycomb(([1, 2, 3, 4]))
    print(x, x.shape)

    x = arraycomb([[0, 64, 128, 192, 256, 320, 384, 448], [0,  64, 128, 192, 256, 320, 384, 448]])
    print(x, x.shape)
