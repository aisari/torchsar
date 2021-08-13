#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-25 11:06:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
from progressbar import *
import torchsar as ts
import torch as th
from torchsar.utils.const import *
import matplotlib.pyplot as plt
import time


def spotlight_width(H, BWa, Lsar):
    return 2. * H * np.tan(BWa / 2.) - Lsar / (2. * H)


def phase_correction(x, phi, nsub=None, axis=-2, ftshift=True):

    if th.is_complex(x):
        # N, Na, Nr = x.size(0), x.size(-2), x.size(-1)
        cplxflag = True
    elif x.size(-1) == 2:
        # N, Na, Nr = x.size(0), x.size(-3), x.size(-2)
        x = th.view_as_complex(x)
        cplxflag = False
    else:
        raise TypeError('x is complex and should be in complex or real represent formation!')
    if axis == -2 or axis == 1:  # azimuth
        axis = -2
    if axis == -1 or axis == 2:  # range
        axis = -1

    if x.dim() == 2:
        x = x.unsqueeze(0)
    if phi.dim() == 1:
        phi = phi.unsqueeze(0)
    if phi.size(-1) != 1:
            phi = phi.unsqueeze(-1)
            phi = phi.unsqueeze(-1)

    N, Na, Nr = x.size()
    Nx = x.size(axis)
    pshape = [1, 1, 1]
    pshape[0] = N
    pshape[axis] = Nx
    phi = phi.reshape(pshape)

    if nsub is None:
        nsub = Nx

    # X = ts.fft(SI, axis=axis, shift=ftshift)
    # X = X * np.exp(-1j * phi)
    # SI = ts.ifft(X, axis=axis, shift=ftshift)
    xc = x.clone().detach()
    d = xc.dim()
    for n in range(0, Nx, nsub):
        idx = ts.sl(d, axis, range(n, n + nsub))
        xsub = x[idx]
        phisub = phi[idx]
        Xsub = ts.fft(xsub, axis=axis, shift=ftshift)
        xc[idx] = ts.ifft(Xsub * th.exp(-1j * phisub), axis=axis, shift=ftshift)
    if not cplxflag:
        xc = th.view_as_real(xc)
    return xc


def pgaf_sm_1iter(x, windb=None, est='ML', deg=4, axis=-2, isrmlpe=False, iscrct=False, isplot=False):
    r"""perform phase gradient autofocus 1 iter

    Perform phase gradient autofocus 1 iter as described in [1].

    revised for stripmap SAR

    - [1] C.V. Jakowatz, D.E. Wahl, P.H. Eichel, D.C. Ghiglia, P.A. Thompson,
    {\em Spotlight-mode Synthetic Aperture Radar: A Signal Processing
    Approach.} Springer, 1996.

    - [2] D.E. Wahl, P.H. Eichel, D.C. Ghiglia, C.V. Jakowatz, "Phase
    gradient autofocus-a robust tool for high resolution SAR phase
    correction," IEEE Trans. Aero. & Elec. Sys., vol. 30, no. 3, 1994.

    Args:
        x (Tensor): Complex SAR image :math:`{\bm X}\in {\mathbb C}^{N×N_a×N_r}`, where :math:`N` is the batchsize.
        windb (None or float, optional): Cutoff for window (the default is None, which use the mean as the cutoff.)
        est (str, optional): Estimator, ``'ML'`` for Maximum Likelihood estimation, ``'LUMV'`` for Linear Unbiased Minimum Variance estimation (the default is 'ML')
        deg (int): The degree of polynominal
        axis (int, optional): Autofocus axis.
        isrmlpe (bool, optional): Is remove linear phase error? (the default is False)
        iscrct (bool, optional): Is corrected image? (the default is False)
        isplot (bool, optional): Is plot estimated phase error? (the default is False)

    Returns:
        xc (Tensor): Corrected SAR image :math:`{\bm Y}\in {\mathbb C}^{N×N_a×N_r}`, only if :attr:`iscrct` is ``True``, xc is returned.
        phi (Tensor): Estimated phase error :math:`\phi\in {\mathbb R}^{N×N_a}`.

    Raises:
        TypeError: :attr:`x` is complex and should be in complex or real represent formation!
        ValueError: Not supported estimator! Supported are ``'ML'`` and ``'LUMV'``.

    """

    raise TypeError('Not opened yet!')


def pgaf_sm(SI, nsar, nsub=None, windb=None, est='ML', deg=4, niter=None, tol=1.e-6, isplot=False, islog=False):
    r"""Phase gradient autofocus for stripmap SAR.

    Phase gradient autofocus for stripmap SAR.

    Args:
        SI (Tensor): Complex SAR image :math:`N_a×N_r`.
        nsar (int): Number of synthetic aperture pixels.
        nsub (int, optional): Number of sub-aperture pixels. (the default is :math:`{\rm min}{N_{sar}, N_a}`)
        windb (None or float, optional): cutoff for window (the default is None, which use the mean as the cutoff.)
        est (str, optional): estimator, ``'ML'`` for Maximum Likelihood estimation, ``'LUMV'`` for Linear Unbiased Minimum Variance estimation (the default is 'ML')
        deg (int): Polynomial degrees (default 4) to fit the error, once fitted, the term of deg=[0,1] will be removed.
            If :attr:`deg` is None or lower than 2, then we do not fit the error with polynominal and not remove the linear trend.
        niter (int, optional): Maximum iters (the default is None, which means using :attr:`tol` for stopping)
        tol (float, optional): Phase error tolerance. (the default is 1.e-6)
        isplot (bool, optional): Plot estimated phase error or corrected image. (the default is False)
        islog (bool, optional): Print log information? (the default is False)

    Returns:
        SI (Tensor): Corrected SAR image :math:`{\bm X}\in {\mathbb C}^{N_a×N_r}`.
        phi (Tensor): Estimated phase error :math:`{\phi}\in {\mathbb R}^{N_a×1}`.

    Raises:
        TypeError: The input is complex and should be in complex or real represent formation!
        ValueError: For stripmap SAR, processing sub aperture should be smaller than synthetic aperture!
    """

    raise TypeError('Not opened yet!')


if __name__ == '__main__':

    matfile = '/mnt/e/ws/github/psar/psar/examples/imaging/data/ALPSRP020160970_Vr7180_R3.mat'
    SI = ts.loadmat(matfile)['SI']

    SI0 = SI
    # sa, ea, sr, er = 3500, 3500 + 2048, 5500, 5500 + 2048
    sa, ea, sr, er = 4600, 4600 + 1024, 5000, 5000 + 1024
    # sa, ea, sr, er = 3000, 3000 + 512, 5000, 5000 + 512
    # sa, ea, sr, er = 3200, 3200 + 256, 5000, 5000 + 256

    SI = SI[sa:ea, sr:er, 0] + 1j * SI[sa:ea, sr:er, 1]

    SI0 = SI0[sa:ea, sr:er, 0] + 1j * SI0[sa:ea, sr:er, 1]

    deg = 7
    # deg = None
    est = 'ML'
    est = 'LUMV'

    niter = 40

    SI = th.from_numpy(SI)
    SI0 = th.from_numpy(SI0)
    # SI = SI.to('cuda:0')
    SI, pa = ts.pgaf_sm(SI, 6785, nsub=None, windb=None, est=est, deg=deg, niter=niter, tol=1.e-6, isplot=False, islog=False)

    print(pa.shape)
    plt.figure()
    plt.plot(pa[-1])
    plt.grid()
    plt.xlabel('Aperture time (samples)')
    plt.ylabel('Phase (rad)')
    plt.title('Estimated phase error (polynomial degree ' + str(deg) + ')')
    plt.show()

    SI0 = th.abs(SI0)
    SI = th.abs(SI)
    print("ENT:", ts.entropy(SI0))
    print("ENT:", ts.entropy(SI))
    print("MSE", th.sum(SI0 - SI))

    print(SI.shape, SI0.shape)
    SI = ts.mapping(SI)
    SI0 = ts.mapping(SI0)
    print(SI.shape, SI0.shape)

    plt.figure()
    plt.subplot(121)
    plt.imshow(SI0, cmap='gray')
    plt.subplot(122)
    plt.imshow(SI[-1], cmap='gray')
    plt.show()
