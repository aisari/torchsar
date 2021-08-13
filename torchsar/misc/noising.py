#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-18 21:31:56
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th
from torchsar.utils.convert import str2num
from torchsar.base.mathops import nextpow2


def matnoise(mat, noise='wgn', snr=30, peak='maxv'):
    """add noise to an matrix

    Add noise to an matrix (real or complex)

    Args:
        mat (torch.Tensor): Input tensor, can be real or complex valued
        noise (str, optional): type of noise (default: ``'wgn'``)
        snr (float, optional): Signal-to-noise ratio (default: 30)
        peak (None or float, optional): Peak value in input, for complex data, ``peak=[peakr, peaki]``, if None, auto detected, if ``'maxv'``, use the maximum value as peak value. (default)

    Returns:
        (torch.Tensor): Output tensor.
    """

    if th.is_complex(mat):
        cplxflag = True
        if peak is None:
            # peakr = mat.real.abs().max()
            peakr = mat.real.max()
            peaki = mat.imag.max()
            peakr = 2**nextpow2(peakr) - 1
            peaki = 2**nextpow2(peaki) - 1
        if peak == 'maxv':
            # peakr = mat.real.abs().max()
            peakr = mat.real.max()
            peaki = mat.imag.max()
        else:
            peakr, peaki = peak
        mat = th.view_as_real(mat)
        mat[..., 0] = awgn(mat[..., 0], snr=snr, peak=peakr, pmode='db', measMode='measured')
        mat[..., 1] = awgn(mat[..., 1], snr=snr, peak=peaki, pmode='db', measMode='measured')
    else:
        cplxflag = False
        if peak is None:
            # peakr = mat.real.abs().max()
            peakr = mat.real.max()
            peaki = mat.imag.max()
            peakr = 2**nextpow2(peakr) - 1
            peaki = 2**nextpow2(peaki) - 1
        if peak == 'maxv':
            # peakr = mat.real.abs().max()
            peakr = mat[..., 0].max()
            peaki = mat[..., 1].max()
        else:
            peakr, peaki = peak
        if mat.shape[-1] == 2:
            mat[..., 0] = awgn(mat[..., 0], snr=snr, peak=peakr, pmode='db', measMode='measured')
            mat[..., 1] = awgn(mat[..., 1], snr=snr, peak=peaki, pmode='db', measMode='measured')
        else:
            if peak is None:
                # peak = mat.abs().max()
                peak = mat.max()
                peak = 2**nextpow2(peak) - 1
            elif peak == 'maxv':
                peak = mat.max()
            mat = awgn(mat, snr=snr, peak=peak, pmode='db', measMode='measured')

    if cplxflag:
        mat = th.view_as_complex(mat)

    return mat


def imnoise(img, noise='wgn', snr=30, peak=None, fmt='chnllast'):
    """Add noise to image

    Add noise to image

    Args:
        img (torch.Tensor): image aray
        noise (str, optional): noise type (the default is 'wgn', which [default_description])
        snr (float, optional): Signal-to-noise ratio (the default is 30, which [default_description])
        peak (None, str or float): Peak value in input, if None, auto detected (default), if ``'maxv'``, use the maximum value as peak value.
        fmt (str or None, optional): for color image, :attr:`fmt` should be specified with ``'chnllast'`` or ``'chnlfirst'``, for gray image, :attr:`fmt` should be setted to ``None``.

    Returns:
        (torch.Tensor): Images with added noise.

    """

    if peak is None:
        peak = 2**str2num(str(img.dtype), int)[0] - 1.
    elif peak == 'maxv':
        peak = img.max()

    if img.dim() == 2:
        img = awgn(img, snr, peak=peak, pmode='db', measMode='measured')
    elif img.dim() == 3:
        if fmt in ['chnllast', 'ChnlLast']:
            for c in range(img.shape[-1]):
                img[..., c] = awgn(img[..., c], snr, peak=peak, pmode='db', measMode='measured')
        if fmt in ['chnlfirst', 'ChnlFirst']:
            for c in range(img.shape[0]):
                img[c, ...] = awgn(img[c, ...], snr, peak=peak, pmode='db', measMode='measured')
        if fmt is None:  # gray image
            for n in range(img.shape[0]):
                img[n, ...] = awgn(img[n, ...], snr, peak=peak, pmode='db', measMode='measured')
    elif img.dim() == 4:
        if fmt in ['chnllast', 'ChnlLast']:
            for n in range(img.shape[0]):
                for c in range(img.shape[-1]):
                    img[n, :, : c] = awgn(img[n, :, : c], snr, peak=peak, pmode='db', measMode='measured')
        if fmt in ['chnlfirst', 'ChnlFirst']:
            for n in range(img.shape[0]):
                for c in range(img.shape[1]):
                    img[n, c, ...] = awgn(img[n, c, ...], snr, peak=peak, pmode='db', measMode='measured')
    return img


def awgn(sig, snr=30, peak=1, pmode='db', measMode='measured'):
    """AWGN Add white Gaussian noise to a signal.

    Y = AWGN(X,snr) adds white Gaussian noise to X.  The snr is in dB.
    The power of X is assumed to be 0 dBW.  If X is complex, then
    AWGN adds complex noise.

    Args:
        sig (torch.Tensor): Signal that will be noised.
        snr (float, optional): Signal Noise Ratio (the default is 30)
        peak (float, optional): Peak value (the default is 1)
        pmode (str, optional): Power mode ``'linear'``, ``'db'`` (the default is 'db')
        measMode (str, optional): (the default is 'measured', which means auto computed.)

    Returns:
        (torch.Tensor): Images with added noise.

    Raises:
        IOError: No input signal
        TypeError: Input signal shape wrong
    """

    # --- Set default values
    sigPower = 0
    reqsnr = snr
    # print(sig.shape)
    # --- sig
    if sig is None:
        raise IOError('NoInput')
    elif sig.ndim > 2:
        raise TypeError("The input signal must have 2 or fewer dimensions.")
    # --- Check the signal power.
    # This needs to consider power measurements on matrices
    if measMode == 'measured':
        sigPower = th.sum(th.abs(sig) ** 2) / sig.numel()
        if pmode == 'db':
            sigPower = 10 * th.log10(sigPower)

    # print(sig.shape)
    # --- Compute the required noise power
    if pmode == 'linear':
        noisePower = sigPower / reqsnr
    elif pmode == 'db':
        noisePower = sigPower - reqsnr
        pmode = 'dbw'

    # --- Add the noise
    if (th.is_complex(sig)):
        dtype = 'complex'
    else:
        dtype = 'real'

    y = sig + wgn(sig.shape, noisePower, peak, pmode, dtype)
    return y


def wgn(shape, p, peak=1, pmode='dbw', dtype='real', seed=None):
    """WGN Generate white Gaussian noise.

    Y = WGN((M,N),P) generates an M-by-N matrix of white Gaussian noise. P
    specifies the power of the output noise in dBW. The unit of measure for
    the output of the wgn function is Volts. For power calculations, it is
    assumed that there is a load of 1 Ohm.

    Args:
        shape (tulpe): Shape of noising matrix
        p (float): Specifies the power of the output noise in dBW.
        peak (float, optional): Peak value (the default is 1)
        pmode (str, optional): Power mode of the output noise (the default is 'dbw')
        dtype (str, optional): data type, real or complex (the default is 'real', which means real-valued)
        seed (int, optional): Seed for random number generator. (the default is None, which means different each time)

    Returns:
        torch.Tensor: Matrix of white Gaussian noise (real or complex).
    """

    # print(shape)
    if pmode == 'linear':
        noisePower = p
    elif pmode == 'dbw':
        noisePower = 10 ** (p / 10)
    elif pmode == 'dbm':
        noisePower = 10 ** ((p - 30) / 10)

    # --- Generate the noise
    if seed is not None:
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    if dtype == 'complex':
        y = (th.sqrt(peak * noisePower / 2)) * \
            (_func(shape) + 1j * _func(shape))
    else:
        y = (th.sqrt(peak * noisePower)) * _func(shape)
    # print(y)
    return y


def _func(ab):
    if len(ab) == 1:
        n = th.randn(ab[0])
    else:
        n = th.randn(ab[0], ab[1])
    return n
