#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-06 10:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th
import numpy as np
from torchsar.utils.const import *

from torchsar.simulation.geometry import rectangle, disc
from torchsar.utils.image import imresize
from torchsar.base.randomfunc import setseed


def gpts(scene, n=None, nmax=100, seed=None, device='cpu', verbose=False):
    r"""Generates number of point targets.

    Generates number of point targets.

    Parameters
    ----------
    scene : list or tuple
        Scene Area, [xmin, xmax, ymin, ymax]
    n : int or None
        number of targets, if None, randomly chose from 0 to :attr:`nmax`
    nmax : int
        maximum number of targets, default is 100.
    seed : int, optional
        random seed (the default is None, which different seed every time.)
    device : str, optional
        device
    verbose : bool, optional
        show more log info (the default is False, which means does not show)

    Returns
    -------
    targets : tensor
        [tg1, tg2, ..., tgn], tgi = [x,y]
    """

    if verbose:
        print(scene)
    (xmin, xmax, ymin, ymax) = scene

    setseed(seed, 'torch')

    if n is None:
        n = th.randint(0, nmax, (1,), device=device).item()

    x = th.rand((n, 1), device=device) * (xmax - xmin) + xmin
    y = th.rand((n, 1), device=device) * (ymax - ymin) + ymin

    vx = th.zeros((n, 1), device=device)
    vy = th.zeros((n, 1), device=device)
    ax = th.zeros((n, 1), device=device)
    ay = th.zeros((n, 1), device=device)
    rcs = th.rand((n, 1), device=device)
    # print(rcs)
    # rcs = np.ones(n)
    # rcs[0] = 1

    targets = th.cat((x, y, vx, vy, ax, ay, rcs), 1)

    if verbose:
        print(targets)

    return targets


def gdisc(scene, n=None, nmax=100, centers=None, radius=None, amps=None, dx=None, dy=None, seed=None, verbose=False):
    r"""Generates number of Disc targets.

    Generates Disc targets.

    Parameters
    ----------
    scene : list or tuple
        Scene Area, [xmin,xmax,ymin,ymax]
    n : int
        number of disks
    centers : list, optional
        disk centers (the default is None, which generate randomly)
    radius : list, optional
        disk radius [] (the default is None, which generate randomly)
    amps : list, optional
        amplitudes (the default is None, which generate randomly)
    dx : float, optional
        resolution in range (default: {1 / (xmax-xmin)})
    dy : float, optional
        resolution in azimuth (default: {1 / (ymax-ymin)})
    seed : int or None, optional
        random seed (the default is None, which different seed every time.)
    verbose : bool, optional
        show more log info (the default is False, which means does not show)

    Returns
    -------
    targets : lists
        [tg1, tg2, ..., tgn], tgi = [x,y]
    """

    if verbose:
        print(scene)
    (xmin, xmax, ymin, ymax) = scene

    if seed is not None:
        th.seed(seed)

    if n is None:
        n = th.randint(0, nmax, (1,)).item()

    if centers is None:
        x0 = th.randint(xmin, xmax, n) * 1.0
        y0 = th.randint(ymin, ymax, n) * 1.0
    else:
        x0 = centers[0]
        y0 = centers[1]

    n = len(x0)

    if radiusMax is None:
        radiusMax = 20
    if radiusMin is None:
        radiusMin = 1

    if radius is None:
        radius = th.randint(radiusMin, radiusMax, n)
    if amps is None:
        amps = th.rand(n)

    r = radius

    targets = disc(scene, x0[0], y0[0], r[0], a=amps[0], dx=dx, dy=dy, verbose=False)
    for n in range(1, n):
        target = disc(scene, x0[n], y0[n], r[n], a=amps[n], dx=dx, dy=dy, verbose=False)
        targets = np.concatenate((targets, target), axis=0)

    if verbose:
        print(targets)

    return targets


def grectangle(scene, n, amps=None, h=None, w=None, dx=None, dy=None, seed=None, verbose=False):
    """Generates number of rectangle targets.

    Generates number of rectangle targets.

    Parameters
    ----------
    scene : list or tuple
        Scene Area, [xmin, xmax, ymin, ymax]
    n : int
        number of rectangles
    amps : list, optional
        amplitudes (the default is None, which generate randomly)
    height : list, optional
        height of each rectangle (the default is None, which generate randomly)
    width : list, optional
        width of each rectangle (the default is None, which generate randomly)
    dx : float, optional
        resolution in range (default: {1 / (xmax-xmin)})
    dy : float, optional
        resolution in azimuth (default: {1 / (ymax-ymin)})
    seed : int, optional
        random seed (the default is None, which different seed every time.)
    verbose : bool, optional
        show more log info (the default is False, which means does not show)

    Returns
    -------
    targets : tensor
        [tg1, tg2, ..., tgn], tgi = [x,y]
    """

    if verbose:
        print(scene)
    (xmin, xmax, ymin, ymax) = scene

    if seed is not None:
        setseed(seed, 'torch')

    if amps is None:
        amps = th.rand(n)

    x0 = th.randint(xmin, xmax, n) * 1.0
    y0 = th.randint(ymin, ymax, n) * 1.0
    targets = rectangle(scene, x0[0], y0[0], h, w, a=amps[0], dx=None, dy=None, verbose=False)
    for n in range(1, n):
        target = rectangle(scene, x0[n], y0[n], h, w, a=amps[
                           n], dx=None, dy=None, verbose=False)
        targets = np.concatenate((targets, target), axis=0)

    if verbose:
        print(targets)

    return targets


def dsm2tgs(dsm, scene, bg=0, dsize=None, tsize=None, device='cpu'):
    r"""convert digital scene matrix (or digital surface model?) to targets

    Convert digital scene matrix (or digital surface model?) to targets.

    The digital scene matrix (or digital surface model?) has the size of
    :math:`H×W×C_1`, where :math:`H` is the height (Y axis) of the matrix,
    :math:`W` is the width (X axis) of the matrix and :math:`C_1` is the number
    of attributes (without (x, y) coordinate), such as target height(Z axis),
    target velocity, target class and target intensity.


    Parameters
    ----------
    dsm : numpy.tensor or torch.Tensor
        Digital scene matrix.
    bg : float, optional
        Background value, 0:black, 1:white (default: {0})
    scene : list or tuple, optional
        Scene area [xmin, xmax, ymin, ymax] (default: {None}, (-W / 2, W / 2, -H / 2, H / 2))
    dsize : list or tuple, optional
        Digital scene matrix size, resize :attr:`dsm` to dsize (default: None, equals to :attr:`dsm`).
    device : str, optional
        Device string, such as ``'cpu'``(cpu), ``'cuda:0'``, ``'cuda:1'``.

    Returns
    -------
    targets : torch.Tensor
        Targets lists or tensor with shape :math:`N × C_2`,
        where, :math:`N` is the number of targets,
        :math:`C_2` is the dimension of target attribute (with (x, y) coordinate).
    """

    if type(dsm) is not th.Tensor:
        dsm = th.tensor(dsm, device=device)
    else:
        dsm = dsm.to(device)

    if dsm.dim() < 2:
        raise ValueError("---The input digital scene matrix should has 2 or more dimensions!")
    elif dsm.dim() == 2:
        dsm = dsm.unsqueeze(-1)

    H, W, C1 = dsm.shape

    Lx = scene[1] - scene[0]
    Ly = scene[3] - scene[2]

    Hs, Ws = int(Ly), int(Lx)

    if dsize is None:
        dsize = [Hs, Ws]

    if len(dsize) < 2:
        raise ValueError("---The height and width of digital scene matrix must be given!")

    gH, gW, _ = dsm.shape

    if dsize[0] != gH or dsize[1] != gW:
        dsm = imresize(dsm, oshape=dsize[0:2], odtype=dsm.dtype, preserve_range=True)

    gH, gW, _ = dsm.shape

    Coff = 2
    tgidx = dsm[..., -1] > bg
    nTgs = tgidx.sum().item()
    if tsize is None:
        tsize = [nTgs, C1 + Coff]
    if type(tsize) is int:
        tsize = [nTgs, tsize]

    xmin, xmax, ymin, ymax = scene
    xx = th.linspace(xmin, xmax, gW, device=device).reshape(1, gW).repeat(gH, 1)
    yy = th.linspace(ymax, ymin, gH, device=device).reshape(gH, 1).repeat(1, gW)
    targets = th.zeros(tsize, device=device, dtype=dsm.dtype)

    targets[:, 0] = xx[tgidx]
    targets[:, 1] = yy[tgidx]
    targets[:, -C1:] = dsm[tgidx][:, -C1:]

    # n = 0
    # for h in range(gH):
    #     for w in range(gW):
    #         if dsm[h, w, -1] > 0:
    #             targets[n, -C1:] = dsm[h, w, -C1:]
    #             targets[n, 0] = xx[h, w]
    #             targets[n, 1] = yy[h, w]
    #             n += 1
    # tend = time.time()

    return targets


def tgs2dsm(tgs, scene, bg=0, dsize=None, device='cpu'):
    r"""targets to digital scene matrix (or digital surface model?)

    Convert targets to digital scene matrix (or digital surface model?).

    The digital scene matrix (or digital surface model?) has the size of
    :math:`H×W×C_1`, where :math:`H` is the height (Y axis) of the matrix,
    :math:`W` is the width (X axis) of the matrix and :math:`C_1` is the number
    of attributes (without (x, y) coordinate), such as target height(Z axis),
    target velocity, target class and target intensity.

    Parameters
    ----------
    targets : list or torch.Tensor
        Targets lists or tensor with shape :math:`N × C_2`,
        where, :math:`N` is the number of targets,
        :math:`C_2` is the dimension of target attribute (with (x, y) coordinate).
    scene : list or tuple, optional
        Scene area [xmin, xmax, ymin, ymax]
    bg : float, optional
        Background value, 0:black, 1:white (default: {0})
    dsize : list or tuple, optional
        Size of digital scene matrix (H, W) (default: None, the scene is discretized
        with 1m)

    Returns
    -------
    dsm : torch.Tensor
        Digital scene matrix.
    """

    if type(tgs) is not th.Tensor:
        targets = th.tensor(tgs, device=device)
    else:
        targets = tgs.clone().detach()

    index = th.zeros(targets.shape[0:2], device=device, dtype=th.int16)
    targets = targets.to(device)

    N, C2 = targets.shape

    Lx = scene[1] - scene[0]
    Ly = scene[3] - scene[2]

    Coff = 0
    if dsize is None:
        dsize = [int(Ly / 1.), int(Lx / 1.), C2 - Coff]
    else:
        if len(dsize) < 2:
            raise ValueError("---The height and width of digital scene matrix must be given!")
        if dsize[0] is None:
            dsize[0] = int(Ly / 1.)
        if dsize[1] is None:
            dsize[1] = int(Lx / 1.)
        if len(dsize) > 2 and dsize[-1] is None:
            dsize[-1] = C2 - Coff

    dsm = th.ones(dsize, device=device)

    dsm *= bg

    H, W = dsize[0:2]

    dx = Lx / (W * 1.0)
    dy = Ly / (H * 1.0)

    index[:, 0] = (targets[:, 0] - scene[0]) / dx - 1
    index[:, 1] = (scene[3] - targets[:, 1]) / dy - 1

    if len(dsize) == 2:
        for idx, target in zip(index, targets):
            dsm[idx[1], idx[0]] = target[-1]  # W:x, H:y
    else:
        for idx, target in zip(index, targets):
            dsm[idx[1], idx[0], -dsize[2]:] = target[-dsize[2]:]  # W:x, H:y

    return dsm


if __name__ == '__main__':

    targets = [
        [0, 0, 0, 0, 0, 0, 1.],
        [-1, -1, 0, 0, 0, 0, .2],
        [2, -1, 0, 0, 0, 0, .3],
        [1, 2, 0, 0, 0, 0, .8],
        [-1, 1, 0, 0, 0, 0, .4],
    ]

    dsm = tgs2dsm(targets, scene=(-4, 4, -4, 4), bg=0, dsize=(4, 4, 5))
    print(dsm.shape)
    # print(dsm[3, 3, :])
    # print(dsm[5, 5, :])

    targets = dsm2tgs(dsm, scene=(-4, 4, -4, 4), bg=0, dsize=(4, 4, 5), tsize=7)
    print(targets)
