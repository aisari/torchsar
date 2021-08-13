#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import torch as th
import torchsar as ts
from progressbar import *


def __entropy(X):

    if th.is_complex(X):  # N-Na-Nr
        X = (X * X.conj()).real
    elif X.size(-1) == 2:  # N-Na-Nr-2
        X = th.sum(X.pow(2), axis=-1)

    logfunc, axis = th.log2, (1, 2)
    P = th.sum(X, axis=axis, keepdims=True)
    p = X / (P + ts.EPS)
    S = -th.sum(p * logfunc(p + ts.EPS), axis)
    return th.mean(S)


def meaf_ssa_sm(x, niter=10, delta=None, toli=1e-2, tolo=1e-2, ftshift=True, islog=False):
    r"""Stage-by-Stage minimum entropy

    [1] Morrison Jr, Robert Lee; Autofocus, Entropy-based (2002): Entropy-based autofocus for synthetic aperture radar.

    Parameters
    ----------
    x : Tensor
       Corrupt complex SAR image. N-Na-Nr(complex) or N-Na-Nr-2(real)
    niter : int, optional
       The number of iteration (the default is 10)
    delta : {float or None}, optional
       The change step (the default is None (i.e. PI))
    toli : int, optional
       Tolerance error for inner loop (the default is 1e-2)
    tolo : int, optional
       Tolerance error for outer loop (the default is 1e-2)
    ftshift : bool, optional
       Shift the zero frequency to center? (the default is True)
    islog : bool, optional
       Print log information? (the default is False)
    """

    raise TypeError('Not opened yet!')


def meaf_sm(x, phi=None, niter=10, tol=1e-4, eta=0.1, method='N-MEA', selscat=False, axis=-2, ftshift=True, islog=False):
    r"""Entropy based autofocus

    Minimum-Entropy based Autofocus (MEA)

    Args:
        x (Tensor): complex image with shape :math:`N×N_a×N_r` or :math:`N×N_a×N_r×2`
        phi (Tensor, optional): initial value of :math:`\phi` (the default is None, which means zeros)
        niter (int, optional): number of iterations (the default is 10)
        tol (float, optional): Error tolerance.
        eta (float, optional): Learning rate.
        method (str, optional): method used to update the phase error
            - ``'FP-MEA'`` --> Fix Point
            - ``'CD-MEA'`` --> Coordinate Descent
            - ``'SU-MEA'`` --> Simultaneous Update
            - ``'N-MEA'`` --> Newton, see [2], [1] has problem when used to small image
            - ``'SN-MEA'`` --> Simplified Newton
            - ``'MN-MEA'`` --> Modified Newton
        selscat (bool, optional): Select brighter scatters.
        axis (int, optional): Compensation axis.
        ftshift (bool, optional): Does shift zero frequency to center?
        islog (bool, optional): Does print log info.


        see [1] Zhang S , Liu Y , Li X . Fast Entropy Minimization Based Autofocusing Technique for ISAR Imaging[J]. IEEE Transactions on Signal Processing, 2015, 63(13):3425-3434.
            [2] Zeng T , Wang R , Li F . SAR Image Autofocus Utilizing Minimum-Entropy Criterion[J]. IEEE Geoence & Remote Sensing Letters, 2013, 10(6):1552-1556.

    Returns:
        (Tensor): Description
    """

    raise TypeError('Not opened yet!')


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    matfile = '/mnt/e/ws/github/psar/psar/examples/imaging/data/ALPSRP020160970_Vr7180_R3.mat'
    SI = ts.loadmat(matfile)['SI']

    SI0 = SI
    # sa, ea, sr, er = 3500, 3500 + 2048, 5500, 5500 + 2048
    sa, ea, sr, er = 4600, 4600 + 1024, 5000, 5000 + 1024
    sa, ea, sr, er = 3000, 3000 + 512, 5000, 5000 + 512
    # sa, ea, sr, er = 3200, 3200 + 256, 5000, 5000 + 256

    SI = SI[sa:ea, sr:er, 0] + 1j * SI[sa:ea, sr:er, 1]

    SI0 = SI0[sa:ea, sr:er, 0] + 1j * SI0[sa:ea, sr:er, 1]

    deg = 7
    method, niter = 'N-MEA', 160
    # method, niter = 'FP-MEA', 400

    SI = th.from_numpy(SI)
    SI0 = th.from_numpy(SI0)
    # SI = SI.to('cuda:0')
    SI, pa = ts.meaf_sm(SI, phi=None, niter=niter, tol=1e-4, eta=0.1, method=method, selscat=False, ftshift=True, islog=True)

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
