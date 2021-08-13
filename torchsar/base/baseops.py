#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


import torch as th
import copy


def dmka(D, Ds):
    """Multi-key value assign

    Multi-key value assign

    Parameters
    ----------
    D : dict
        Main-dict.
    Ds : dict
        Sub-dict.
    """

    for k, v in Ds.items():
        D[k] = v
    return D


def cat(shapes, axis=0):
    r"""Concatenates

    Concatenates the given sequence of seq shapes in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    Parameters
    ----------
    shapes : tuple or list
        (shape1, shape2, ...)
    axis : int, optional
        specify the concatenated axis (the default is 0)

    Returns
    -------
    tulpe or list
        concatenated shape

    Raises
    ------
    ValueError
        Shapes are not consistent in axises except the specified one.
    """
    x = 0
    s = copy.copy(shapes[0])
    s = list(s)
    for shape in shapes:
        for ax in range(len(s)):
            if (ax != axis) and (s[ax] != shape[ax]):
                raise ValueError("All tensors must either have \
                    the same shape (except in the concatenating dimension)\
                     or be empty.")
        x += shape[axis]
        # print(x)
    print(s, x)
    s[axis] = x
    return s


def concat2(X1, X2, axis):
    r"""concat2

    concatenate [X1, X2] in aixs direction

    Parameters
    ----------
    X1 : Tensor
        the first torch tensor for concatenating
    X2 : Tensor
        the second torch tensor for concatenating
    axis : int
        concatenating axis

    Returns
    -------
    Tensor
        concatenated tensors
    """

    if X1 is None or X1 is []:
        return X2
    else:
        return th.cat((X1, X2), axis)


if __name__ == '__main__':
    import torchtool as tht
    import torch as th

    D = {'a': 1, 'b': 2, 'c': 3}
    Ds = {'b': 6}
    print(D)
    dmka(D, Ds)
    print(D)

    x = th.randn(2, 3)
    xs = x.shape
    xs = list(xs)
    print(xs)
    print('===cat')
    print(x.size())
    print('---Theoretical result')

    ys = tht.cat((xs, xs, xs), 0)
    print(ys)

    ys = tht.cat((xs, xs, xs), 1)
    print(ys)
    print('---Torch result')

    y = th.cat((x, x, x), 0)
    print(y.size())
    y = th.cat((x, x, x), 1)
    print(y.size())
