#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-26 09:51:56
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


from __future__ import division, print_function, absolute_import
import math
import torch as th
from ..utils.const import *
from ..dsp import normalsignals as sig
from torchsar.sharing.antenna_pattern import antenna_pattern_azimuth


def sarmodel(pdict, mod='2D1', gdshape=None, device='cpu', islog=False):
    r"""model sar imaging process.

    SAR imaging model

    Parameters
    ----------
    pdict: dict
        SAR platform parameters
    mod: str, optional
        mod type(default: '2D1')

            if mod is '2D1':
                the model will be in vector mode:
                    s = A      g
                    MNx1      MNxHW  HWx1(M: Na, N: Nr, MN << HW)
            if mod is '2D2':
                the model will be in mat mode:
                    S = Aa     G     Be
                    MxN       MxH    HxW   WxN(M: Na, N: Nr, M << H, N << W)
            if mod is '1Da':
                    S = A      G
                    MxW       MxH    HxW(M: Na, N: Nr, M << H)
            if mod is '1Dr':
                    S'    =   A      G'
                    NxH       NxW    WxH(M: Na, N: Nr, N << W)

    gdshape: tuple, optional
        discrete scene size of G (default: None, sarplat.acquisition['SceneArea'], dx=dy=1)
    device: str
        device, default is 'cpu'
    islog : bool
        show log info (default: True)

    Returns
    -------
    A : torch tensor
        Imaging mapping matrix.
    """
    raise TypeError('Not opened yet!')


def __computeRmn(ta, tr, H, V, Xc, Yc, SA, gH, gW):
    r"""compute range at g(i, j)

    compute range at g(i, j)

    Arguments
    ------------
    m int
        current y coordinate of H
    n int
        current x coordinate of W
    ta {time in azimuth}
        azimuth time
    tr numpy array
        range time
    H int
        height of SAR platform
    V float
        velocity of SAR platform
    Yc float
        center coordinate in Y axis
    Xc float
        center coordinate in X axis
    SA list
        scene area: [xmin, xmax, ymin, ymax]
    H int
        height of scene(Y)
    W int
        width of scene(X)
    """

    xmin = SA[0] + Xc
    xmax = SA[1] + Xc
    ymin = SA[2] + Yc
    ymax = SA[3] + Yc

    yy = th.linspace(ymin, ymax, gH)
    xx = th.linspace(xmin, xmax, gW)
    ys, xs = th.meshgrid(yy, xx)

    R = th.sqrt(xs ** 2 + (ys - V * ta) ** 2 + H ** 2)  # [gY, gX]
    R = R.reshape(1, -1)
    ys = ys.reshape(1, -1)
    xs = xs.reshape(1, -1)
    return R, xs, ys


def load_sarmodel(datafile, mod='AinvA'):
    r"""load sarmodel file

    load sarmodel file (``.pkl``)

    Parameters
    ----------
    datafile : str
        Model data file path.
    mod : str, optional
        Specify load which variable, ``A``, ``invA``, ``AinvA``
        (the default is 'AinvA', which :math:`\bm A`, :math:`{\bm A}^{-1}` )

    Returns
    -------
    A : numpy array
        Imaging mapping matrix.
    invA : numpy array
        Inverse of imaging mapping matrix.

    Raises
    ------
    ValueError
        wrong mod
    """
    print("===reading model file: ", datafile, "...")
    if datafile != "":
        # get map
        f = open(datafile, 'rb')
        # for python2
        if sys.version_info < (3, 1):
            if mod == 'A':
                A = pkl.load(f)
                f.close()
                return A
            if mod == 'invA':
                pkl.load(f)
                invA = pkl.load(f)
                f.close()
                return invA
            if mod == 'AinvA':
                A = pkl.load(f)
                invA = pkl.load(f)
                f.close()
                return A, invA
            if mod == 'AAH':
                A = pkl.load(f)
                pkl.load(f)
                AH = pkl.load(f)
                f.close()
                return A, AH
            if mod == 'AinvAAH':
                A = pkl.load(f)
                invA = pkl.load(f)
                AH = pkl.load(f)
                f.close()
                return A, invA, AH
            f.close()
            raise ValueError("mod: 'A', 'invA', 'AinvA', 'AAH', 'AinvAAH'")

        # for python3
        else:
            if mod == 'A':
                A = pkl.load(f, encoding='latin1')
                f.close()
                return A
            if mod == 'invA':
                pkl.load(f, encoding='latin1')
                invA = pkl.load(f, encoding='latin1')
                f.close()
                return invA
            if mod == 'AinvA':
                A = pkl.load(f, encoding='latin1')
                invA = pkl.load(f, encoding='latin1')
                f.close()
                return A, invA
            if mod == 'AAH':
                A = pkl.load(f, encoding='latin1')
                pkl.load(f, encoding='latin1')
                AH = pkl.load(f, encoding='latin1')
                f.close()
                return A, AH
            if mod == 'AinvAAH':
                A = pkl.load(f, encoding='latin1')
                invA = pkl.load(f, encoding='latin1')
                AH = pkl.load(f, encoding='latin1')
                f.close()
                return A, invA, AH
            f.close()
            raise ValueError("mod: 'A', 'invA', 'AinvA', 'AAH', 'AinvAAH'")
    else:
        return None


def save_sarmodel(A, invA=None, AH=None, datafile='./model.pkl'):
    r"""save model mapping matrix

    save model mapping matrix to a file.


    Parameters
    ----------
    A : numpy array
        Imaging mapping matrix
    invA : numpy array, optional
        Moore - Penorse inverse of A(default: {None})
    AH : numpy array, optional
        The Hermite :math:`{\bm A}^H` of the Imaging mapping matrix :math:`\bm A`
        (the default is None, which does not store)
    datafile : str, optional
        save file path(default: {'./model.pkl'})
    """

    f = open(datafile, 'wb')

    pkl.dump(A, f, 0)
    if invA is not None:
        pkl.dump(invA, f, 0)
    if AH is not None:
        pkl.dump(AH, f, 0)
    f.close()

