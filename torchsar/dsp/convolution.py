#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
from torchsar.dsp.ffts import padfft, fft, ifft
from torchsar.base.mathops import nextpow2, ebemulcc
from torchsar.base.arrayops import cut


def cutfftconv1(y, nfft, Nx, Nh, shape='same', axis=0, ftshift=False):
    r"""Throwaway boundary elements to get convolution results.

    Throwaway boundary elements to get convolution results.

    Parameters
    ----------
    y : Tensor
        array after ``iff``.
    nfft : int
        number of fft points.
    Nx : int
        signal length
    Nh : int
        filter length
    shape : str
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid convolution output``
        3. ``'full' --> full convolution output``, :math:`N_x+N_h-1`
        (the default is 'same')
    axis : int
        convolution axis (the default is 0)
    ftshift : bool
        whether to shift zero the frequency to center (the default is False)

    Returns
    -------
    y : Tensor
        array with shape specified by :attr:`same`.
    """

    raise TypeError('Not opened yet!')


def fftconv1(x, h, axis=0, nfft=None, shape='same', ftshift=False, eps=None):
    """Convolution using Fast Fourier Transformation

    Convolution using Fast Fourier Transformation.

    Parameters
    ----------
    x : Tensor
        data to be convolved.
    h : Tensor
        filter array
    shape : str, optional
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid convolution output``
        3. ``'full' --> full convolution output``, :math:`N_x+N_h-1`
        (the default is 'same')
    axis : int, optional
        convolution axis (the default is 0)
    nfft : int, optional
        number of fft points (the default is :math:`2^nextpow2(N_x+N_h-1)`),
        note that :attr:`nfft` can not be smaller than :math:`N_x+N_h-1`.
    ftshift : bool, optional
        whether shift frequencies (the default is False)
    eps : {None or float}, optional
        x[abs(x)<eps] = 0 (the default is None, does nothing)

    Returns
    -------
    y : Tensor
        Convolution result array.

    """

    raise TypeError('Not opened yet!')


if __name__ == '__main__':
    import torchsar as ts
    import psar as ps
    import torch as th

    shape = 'same'
    ftshift = False
    # ftshift = True
    x_np = np.array([1, 2, 3, 4, 5])
    h_np = np.array([1 + 2j, 2, 3, 4, 5, 6, 7])

    x_th = th.tensor(x_np)
    h_th = th.tensor(h_np)
    x_th = th.stack([x_th, th.zeros(x_th.size())], dim=-1)
    h_th = th.stack([h_th.real, h_th.imag], dim=-1)

    y1 = ps.fftconv1(x_np, h_np, axis=0, Nfft=None, shape=shape, ftshift=ftshift)
    y2 = ts.fftconv1(x_th, h_th, axis=0, nfft=None, shape=shape, ftshift=ftshift)

    y2 = th.view_as_complex(y2)
    y2 = y2.cpu().numpy()

    print(y1)
    print(y2)
    print(np.sum(np.abs(y1 - y2)), np.sum(np.angle(y1) - np.angle(y2)))
