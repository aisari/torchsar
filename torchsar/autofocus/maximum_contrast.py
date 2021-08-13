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


def mcaf_sm(x, phi=None, niter=10, tol=1e-4, eta=0.1, method='N-MCA', selscat=False, axis=-2, ftshift=True, islog=False):
    r"""Contrast based autofocus

    Maximum-Contrast based Autofocus (MCA)

    Parameters
    ----------
    x : Tensor
        complex image with shape :math:`N×N_a×N_r` or :math:`N×N_a×N_r×2`
    phi : float, optional
        initial value of :math:`\phi` (the default is None, which means zeros)
    niter : int, optional
        number of iterations (the default is 10)
    method : str, optional
        method used to update the phase error
        - ``'FP-MCA'`` --> Fix Point
        - ``'CD-MCA'`` --> Coordinate Descent
        - ``'SU-MCA'`` --> Simultaneous Update
        - ``'N-MCA'`` --> Newton, see [2], [1] has problem when used to small image
        - ``'SN-MCA'`` --> Simplified Newton
        - ``'MN-MCA'`` --> Modified Newton
    selscat : bool, optional
        Select brighter scatters.
    axis : int, optional
        Compensation axis.
    ftshift :  bool, optional
        Does shift zero frequency to center?
    islog :  bool, optional
        Does print log info.

    """

    raise TypeError('Not opened yet!')


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    matfile = '/mnt/e/ws/github/psar/psar/examples/imaging/data/ALPSRP020160970_Vr7180_R3.mat'
    SI = ts.loadmat(matfile)['SI']

    SI0 = SI
    # sa, ea, sr, er = 3500, 3500 + 2048, 5500, 5500 + 2048
    # sa, ea, sr, er = 4600, 4600 + 1024, 5000, 5000 + 1024
    sa, ea, sr, er = 3000, 3000 + 512, 5000, 5000 + 512
    # sa, ea, sr, er = 3200, 3200 + 256, 5000, 5000 + 256

    SI = SI[sa:ea, sr:er, 0] + 1j * SI[sa:ea, sr:er, 1]

    SI0 = SI0[sa:ea, sr:er, 0] + 1j * SI0[sa:ea, sr:er, 1]

    deg = 7
    method, niter = 'N-MCA', 300

    SI = th.from_numpy(SI)
    SI0 = th.from_numpy(SI0)
    # SI = SI.to('cuda:0')
    SI, pa = ts.mcaf_sm(SI, phi=None, niter=niter, tol=1e-4, eta=0.1, method=method, selscat=False, ftshift=True, islog=True)

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
