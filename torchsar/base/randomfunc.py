#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import random
import torch as th
import numpy as np
from torchsar.base.arrayops import arraycomb


def setseed(seed=None, target='torch'):
    r"""set seed

    Set numpy / random / torch / torch.random / torch.cuda seed.

    Parameters
    ----------
    seed : int or None, optional
        seed for random number generator (the default is None)
    target : str, optional
        - ``'numpy'``: ``np.random.seed(seed)``
        - ``'random'``: ``random.seed(seed)``
        - ``'torch'``: ``torch.manual_seed(seed)`` (default)
        - ``'torch.random'``: ``torch.random.manual_seed(seed)``
        - ``'cuda'``: ``torch.cuda.manual_seed(seed)``
        - ``'cudaall'``: ``torch.cuda.manual_seed_all(seed)``

    """

    if target in ['numpy', 'np']:
        np.random.seed(seed)
    if target in ['random', 'rand']:
        random.seed(seed)
    if target in ['torch', 'Torch']:
        if seed is not None:
            th.manual_seed(seed)
    if target in ['torch.random', 'torch random']:
        if seed is not None:
            th.random.manual_seed(seed)
    if target in ['cuda']:
        if seed is not None:
            th.cuda.manual_seed(seed)
    if target in ['cudaall', 'cuda all']:
        if seed is not None:
            th.cuda.manual_seed_all(seed)


def permutation(x):
    r"""permutation function like numpy.random.permutation

    permutation function like numpy.random.permutation

    Parameters
    ----------
    x : Tensor
        inputs, can have any dimensions.

    Returns
    -------
    x : Tensor
        permutated tensor
    """

    if type(x) is not th.Tensor:
        x = th.tensor(x)
    xshape = x.shape
    x = x.flatten()
    n = len(x)
    idx = th.randperm(n)
    return x[idx].reshape(xshape)


def randperm(start, stop, n):
    r"""randperm function like matlab

    genarates diffrent random interges in range [start, stop)

    Parameters
    ----------
    start : {int or list}
        start sampling point

    stop : {int or list}
        stop sampling point

    n : {int, list or None}
        the number of samples (default None, (stop - start))

    see :func:`randgrid`.
    """

    if (n is not None) and (type(n) is not int):
        raise TypeError('The number of samples should be an integer or None!')
    elif n is None:
        n = int(stop - start)

    Ps = []
    starts = [start] if type(start) is int else start
    stops = [stop] if type(stop) is int else stop
    for start, stop in zip(starts, stops):
        P = permutation(range(start, stop, 1))[0:n]
        Ps.append(P)
    if len(starts) == 1:
        Ps = Ps[0]
    return Ps


def randperm2d(H, W, number, population=None, mask=None):
    r"""randperm 2d function

    genarates diffrent random interges in range [start, end)

    Parameters
    ----------
    H : int
        height

    W : int
        width
    number : int
        random numbers
    population : {list or numpy array(1d or 2d)}
        part of population in range(0, H*W)
    """

    if population is None:
        population = th.tensor(range(0, H * W)).reshape(H, W)
    if type(population) is not th.Tensor:
        population = th.tensor(population)
    if mask is not None and th.sum(mask) != 0:
        population = population[mask > 0]

    population = population.flatten()
    population = permutation(population)

    Ph = th.floor(population / W)
    Pw = th.floor(population - Ph * W)
    Ph, Pw = Ph.to(th.int), Pw.to(th.int)

    # print(Pw + Ph * W)
    return Ph[0:number], Pw[0:number]


def randgrid(start, stop, step, shake=0, n=None):
    r"""generates non-repeated uniform stepped random ints

    Generates :attr:`n` non-repeated random ints from :attr:`start` to :attr:`stop`
    with step size :attr:`step`.

    When step is 1 and shake is 0, it works similar to randperm,

    Parameters
    ----------
    start : {int or list}
        start sampling point
    stop : {int or list}
        stop sampling point
    step : {int or list}
        sampling stepsize
    shake : float
        the shake rate, if :attr:`shake` is 0, no shake, (default),
        if positive, add a positive shake, if negative, add a negative.
    n : int or None
        the number of samples (default None, int((stop0 - start0) / step0) * int((stop1 - start1) / step1)...).

    Returns
    -------
        for multi-dimension, return a 2-d tensor, for 1-dimension, return a 1d-tensor.

    Example
    -------

    ::

        import matplotlib.pyplot as plt

        setseed(2021)
        print(randperm(2, 40, 8), ", randperm(2, 40, 8)")
        print(randgrid(2, 40, 1, -1., 8), ", randgrid(2, 40, 1, 8, -1.)")
        print(randgrid(2, 40, 6, -1, 8), ", randgrid(2, 40, 6, 8)")
        print(randgrid(2, 40, 6, 0.5, 8), ", randgrid(2, 40, 6, 8, 0.5)")
        print(randgrid(2, 40, 6, -1, 12), ", randgrid(2, 40, 6, 12)")
        print(randgrid(2, 40, 6, 0.5, 12), ", randgrid(2, 40, 6, 12, 0.5)")

        mask = th.zeros((5, 6))
        mask[3, 4] = 0
        mask[2, 5] = 0

        Rh, Rw = randperm2d(5, 6, 4, mask=mask)

        print(Rh)
        print(Rw)

        y = randperm(0, 8192, 800)
        x = randperm(0, 8192, 800)

        y, x = randgrid([0, 0], [512, 512], [64, 64], [0.0, 0.], 32)
        print(len(y), len(x))

        plt.figure()
        plt.plot(x, y, 'o')
        plt.show()

        y, x = randgrid([0, 0], [8192, 8192], [256, 256], [0., 0.], 400)
        print(len(y), len(x))

        plt.figure()
        plt.plot(x, y, '*')
        plt.show()


    see :func:`randperm`.

    """

    starts = [start] if type(start) is int else start
    stops = [stop] if type(stop) is int else stop
    steps = [step] if type(step) is int else step
    shakes = [shake] if type(shake) is int or type(shake) is float else shake
    if (n is not None) and (type(n) is not int):
        raise TypeError('The number of samples should be an integer or None!')
    elif n is None:
        n = float('inf')
    index = []
    for start, stop, step, shake in zip(starts, stops, steps, shakes):
        shakep = shake if abs(shake) >= 1 and type(shake) is int else int(shake * step)
        x = th.tensor(range(start, stop, step))
        if shakep != 0:
            s = th.randint(0, abs(shakep), (len(x),))
            x = x - s if shakep < 0 else x + s
            x[x >= (stop - step)] = stop - step
            x[x < start] = start
        index.append(x)
    P = arraycomb(index)
    n = min(P.shape[0], n)
    idx = permutation(range(0, P.shape[0], 1))
    P = P[idx[:n], ...]

    if len(starts) == 1:
        P = P.squeeze(1)
        return P
    else:
        return P.t()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    setseed(2021)

    print(randperm(2, 40, 8), ", randperm(2, 40, 8)")
    print(randgrid(2, 40, 1, -1., 8), ", randgrid(2, 40, 1, 8, -1.)")
    print(randgrid(2, 40, 6, -1, 8), ", randgrid(2, 40, 6, 8)")
    print(randgrid(2, 40, 6, 0.5, 8), ", randgrid(2, 40, 6, 8, 0.5)")
    print(randgrid(2, 40, 6, -1, 12), ", randgrid(2, 40, 6, 12)")
    print(randgrid(2, 40, 6, 0.5, 12), ", randgrid(2, 40, 6, 12, 0.5)")

    mask = th.zeros((5, 6))
    mask[3, 4] = 0
    mask[2, 5] = 0

    Rh, Rw = randperm2d(5, 6, 4, mask=mask)

    print(Rh)
    print(Rw)

    y = randperm(0, 8192, 800)
    x = randperm(0, 8192, 800)

    y, x = randgrid([0, 0], [512, 512], [64, 64], [0.0, 0.], 32)
    print(len(y), len(x))

    plt.figure()
    plt.plot(x, y, 'o')
    plt.show()

    y, x = randgrid([0, 0], [8192, 8192], [256, 256], [0., 0.], 400)
    print(len(y), len(x))

    plt.figure()
    plt.plot(x, y, '*')
    plt.show()

    x = th.tensor(range(0, 24))
    x = x.reshape(3, 4, 2)

    print(x, 'before permutation')
    print(permutation(x), 'after permutation')

    x = th.tensor(range(0, 24))

    print(x, 'before permutation')
    print(permutation(x), 'after permutation')
