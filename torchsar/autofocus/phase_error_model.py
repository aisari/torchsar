#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import math
import torch as th
import torchsar as ts


def convert_ppec(cs, degs=None, mode='fftfreq->2fftfreq'):
    r"""convert polynominal phase error

    Convert polynominal phase error.

    Args:
        cs (Tensor): The polynominal coefficients.
        degs (None, optional): The degrees of polynominal.
        mode (str, optional): ``'fftfreq->2fftfreq'`` or ``'2fftfreq->fftfreq'`` (the default is 'fftfeq->2fftfreq')

    Returns:
        Tensor: Converted phase error tensor.

    Raises:
        ValueError: Not support mode.
    """

    if type(cs) is not th.Tensor:
        cs = th.tensor(cs)
    N, M = cs.shape
    if degs is None:
        degs = list(range(2, M + 2))

    ccs = th.zeros_like(cs)
    if mode in ['fftfreq->2fftfreq', 'FFTFREQ->2FFTFREQ']:
        for m, deg in zip(range(M), degs):
            ccs[:, m] = cs[:, m] / (2**deg)
    elif mode in ['2fftfreq->fftfreq', '2FFTFREQ->FFTFREQ']:
        for m, deg in zip(range(M), degs):
            ccs[:, m] = cs[:, m] * (2**deg)
    else:
        raise ValueError('Not support!')
    return ccs


def ppeaxis(n, norm=True, shift=True, mode='fftfreq', dtype=th.float32, device='cpu'):
    r"""Return the sample axis

    Return the sample frequencies axis.

    Parameters
    ----------
    n : int
        Number of samples.
    norm : bool
        Normalize the frequencies.
    shift : bool
        Does shift the zero frequency to center.
    mode : str
        Frequency sampling mode. ``'fftfreq'`` or ``'freq'``, for two times,
        using ``'2fftfreq'`` or ``'2freq'``
    dtype : Tensor
        Data type, default is ``th.float32``.
    device : str
        device string, default is ``'cpu'``.

    Returns
    -------
    torch array
        Frequency array with size :math:`n×1`.
    """
    fs = n
    if mode in ['fftfreq', 'FFTFREQ']:
        return ts.fftfreq(n, fs, norm=norm, shift=shift, dtype=dtype, device=device)
    if mode in ['freq', 'FREQ']:
        return ts.freq(n, fs, norm=norm, shift=shift, dtype=dtype, device=device)
    if mode in ['2fftfreq', '2FFTFREQ']:
        return 2 * ts.fftfreq(n, fs, norm=norm, shift=shift, dtype=dtype, device=device)
    if mode in ['2freq', '2FREQ']:
        return 2 * ts.freq(n, fs, norm=norm, shift=shift, dtype=dtype, device=device)


def polype(c, x, deg=None):
    r"""compute phase error with polynominal model

    compute phase error with polynominal model

    Args:
        c (Tensor, optional): The polynominal coefficients matrix :math:`{\bf P} = [{\bf p}_1, {\bf p}_2, \cdots, {\bf p}_N ]^T ∈{\mathbb R}^{N × (M -1)}`,
            where, :math:`{\bf p}_n = [p_2, \cdots, p_M]^T`).
        x (Tensor, optional): Axis, such as (-1, 1).
        deg (tuple or None, optional): The degree of the coefficients, :math:`[p_{\rm min}, p_{\rm_max}]`, default is ``None``, which means ``[2, size(c, 1) + 1]``

    Returns:
        (Tensor): The phase error tensor.

    Raises:
        TypeError: :attr:`c` should not be None!
    """

    if type(c) is not th.Tensor:
        c = th.tensor(c)
    if type(x) is not th.Tensor:
        x = th.tensor(x)

    if deg is None:
        M = [2, c.size(1) + 1]
    else:
        M = deg
    x = x.reshape(1, th.numel(x))

    if c is not None:
        xs = th.tensor([], device=x.device)
        for m in range(M[0], M[1] + 1):
            xs = th.cat((xs, x ** m), axis=0)
        xs = xs.to(c.device)
        c = th.matmul(c, xs)
    else:
        raise TypeError('c should not be None!')

    return c


def dctpe(c, x, deg=None):
    r"""compute phase error with DCT model

    compute phase error with DCT model

    .. math::
        \phi_{e}(h)=\sum_{p=0}^{P} a(p) \cdot d c t(p) \cdot \cos \left[\frac{\pi\left(2 h_{a}+1\right) p}{2 N}\right]

    where, :math:`a(p)=\left{\begin{array}{ll}1 / \sqrt{N} p=0 \ \sqrt{2 / N} p \neq 0\end{array},\right.` and
    :math:`\left{\begin{array}{l}p=0,1,2, \cdots \cdots \cdots, P \ h_{a}=[0,1,2, \cdots \cdots, N-1]\end{array}\right.`

    Args:
        c (Tensor, optional): The DCT coefficients matrix :math:`{\bf P} = [{\bf p}_1, {\bf p}_2, \cdots, {\bf p}_N ]^T ∈{\mathbb R}^{N × (M + 1)}`,
            where, :math:`{\bf p}_n = [p_0, \cdots, p_M]^T`).
        x (Tensor, optional): Axis, such as :math:`(0, Na)`.
        deg (tuple or None, optional): The degree of the coefficients, :math:`[p_{\rm min}, p_{\rm_max}]`, default is ``None``, which means ``[0, size(c, 1)]``

    Returns:
        (Tensor): The phase error tensor.

    Raises:
        TypeError: :attr:`c` should not be None!
    """

    raise TypeError('Not opened yet!')


def rmlpe(phi, mode='poly', deg=4, ftshift=True):
    r"""remove linear phase error

    Remove linear phase error based on linear fitting, the linear phase error
    will cause image shift.

    Args:
        phi (Tensor): A :math:`N×N_x` phase error tensor with linear trend, where, :math:`N` is the batchsize, :math:`N_x` is the samples numbers in direction x.
        mode (str): model mode, default is polynominal
        deg (int): Polynomial degrees (default 4) to fit the error, once fitted, the term of deg=[0,1] will be removed.
            If :attr:`deg` is None or lower than 2, then we do not fit the error with polynominal and not remove the linear trend.
        ftshift (bool): Shift zero frequency to center when do fft/ifft/fftfreq? (the default is False)

    Returns:
        phi (tensor): Phase without linear trend.
    """

    if phi.dim() == 1:
        phi = phi.unsqueeze(0)  # N-Na
    Nx = phi.size(1)
    if deg is not None and deg >= 2:
        x = ts.ppeaxis(Nx, norm=True, shift=ftshift, mode='2fftfreq', device=phi.device)
        w = ts.polyfit(x, phi, deg=deg)
        w[0, :] = 0.
        w[1, :] = 0.
        phi = ts.polyval(w, x)
        return phi.t()
    else:
        return phi


class PolyPhaseErrorGenerator(object):
    def __init__(self, carange=None, crrange=None, seed=None):
        """Polynominal phase error generator.

        Args:
            carange (None or tuple or list, optional): List of coefficients range of phase error in azimuth direction.
            crrange (None or tuple or list, optional): List of coefficients range of phase error in range direction.
            seed (None or int, optional): The random seed.
        """
        self.carange = carange
        self.crrange = crrange
        self.seed = seed
        self.ma = len(carange[0]) + 1 if carange is not None else None  # order 2-ma
        self.mr = len(crrange[0]) + 1 if crrange is not None else None  # order 2-mr
        if seed is not None:
            ts.setseed(seed)

    def mkpec(self, n, seed=None):
        """Make phase error

        Args:
            n (int): The number of phase errors.
            seed (None or int, optional): The random seed.

        Returns:
            TYPE: Description
        """
        if seed is not None:
            ts.setseed(seed)

        if self.ma is not None:
            ca = th.zeros(n, self.ma - 1)
            for i, lowhigh in enumerate(zip(self.carange[0], self.carange[1])):
                low, high = lowhigh
                ca[:, i] = th.rand(n) * (high - low) + low
        if self.mr is not None:
            cr = th.zeros(n, self.mr - 1)
            for i, lowhigh in enumerate(zip(self.crrange[0], self.crrange[1])):
                low, high = lowhigh
                cr[:, i] = th.rand(n) * (high - low) + low

        if self.ma is not None and self.mr is None:
            return ca
        if self.mr is not None and self.ma is None:
            return cr
        if self.ma is not None and self.mr is not None:
            return ca, cr


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 5
    seed = 2020
    # seed = None
    ftshift = True
    carange = [[-300, -250, -200, -150, -100, -50], [300, 250, 200, 150, 100, 50]]
    crrange = [[-30, -25, -20, -15, -10, -5], [30, 25, 20, 15, 10, 5]]

    carange = [[-300, -300, -300, -300, -300, -300, -300, -300], [300, 300, 300, 300, 300, 300, 300, 300]]

    ppeg = PolyPhaseErrorGenerator(carange, crrange, seed)

    ca, cr = ppeg.mkpec(n=n, seed=None)

    Na, Nr = 256, 256
    xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode='2fftfreq')
    xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode='2fftfreq')
    pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)

    print(ca, ca.shape)
    print(cr, cr.shape)

    print(pa.shape, pr.shape)

    plt.figure()
    plt.subplot(121)
    for i in range(n):
        plt.plot(pa[i])
    plt.subplot(122)
    for i in range(n):
        plt.plot(pr[i])

    plt.show()
