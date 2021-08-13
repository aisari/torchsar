#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-23 19:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th
# import numexpr as ne
from torchsar.dsp.ffts import fft, fftfreq, padfft
from torchsar.utils.const import *
from torchsar.base.mathops import nextpow2
from torchsar.dsp.normalsignals import rect


def chirp_mf_td(K, Tp, Fs, Fc=0., Ns=None, mod='conv', scale=False, device='cpu'):
    """Generates matched filter of chirp signal in time domain

    Generates matched filter of chirp signal in time domain.

    Parameters
    ----------
    K : int
        The chirp rate.
    Tp : float
        The pulse width.
    Fs : float
        The sampling rate.
    Fc : float, optional
        The center frequency.
    Ns : int or None, optional
        The number of samples.
    mod : str, optional
        The mode of filter, ``'conv'`` or ``'corr'``
    scale : bool, optional
        Whether to scale the amplitude of the filter.
    device : str, optional
        Specifies the device to be used for computing.

    Returns
    -------
    tensor
        The matched filter tensor.
    """
    raise TypeError('Not opened yet!')


def chirp_mf_fd(K, Tp, Fs, Fc=0., Nfft=None, mod='way1', win=None, ftshift=False, scale=False, device='cpu'):
    """Summary

    Parameters
    ----------
    K : int
        The chirp rate.
    Tp : float
        The pulse width.
    Fs : float
        The sampling rate.
    Fc : float, optional
        The center frequency.
    Nfft : int or None, optional
        The number of points for doing FFT.
    mod : str, optional
        The mode of matched filter.
    win : tensor or None, optional
        The window function.
    ftshift : bool, optional
        Shift the zero-frequecy in center?
    scale : bool, optional
        Scale the filter?
    device : str, optional
        Specifies the device to be used for computing.

    Returns
    -------
    tensor
        The matched filter tensor.
    """
    raise TypeError('Not opened yet!')


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    Kr = 4.1e+11
    Tp = 37.0e-06
    Br = abs(Kr) * Tp

    alpha = 1.24588  # 1.1-1.4
    Fsr = alpha * Br
    Fc = 5.3e9
    # Fc = 0.

    Tsr = 2.1 * Tp
    Nsr = int(Fsr * Tsr)
    tr = th.linspace(-Tsr / 2., Tsr / 2., Nsr)
    fr = th.linspace(-Fsr / 2., Fsr / 2., Nsr)
    t = th.linspace(-Tsr / 2., Tsr / 2., Nsr)

    Sm1, t = chirp_mf_td(Kr, Tp, Fsr, Fc=Fc, Ns=Nsr, mod='conv')
    Sm2, t = chirp_mf_td(Kr, Tp, Fsr, Fc=Fc, Ns=Nsr, mod='corr')

    f = th.linspace(-Fsr / 2., Fsr / 2., len(Sm1))
    Ym1 = fft(Sm1, axis=0, shift=True)

    f = th.linspace(-Fsr / 2., Fsr / 2., len(Sm2))
    Ym2 = fft(Sm2, axis=0, shift=True)

    plt.figure(1)
    plt.subplot(221)
    plt.plot(t * 1e6, th.real(Sm1))
    plt.plot(t * 1e6, th.abs(Sm1))
    plt.grid()
    plt.legend(['Real part', 'Amplitude'])
    plt.title('Convolution matched filter')
    plt.xlabel(r'Time/$\mu s$')
    plt.ylabel('Amplitude')
    plt.subplot(222)
    plt.plot(t * 1e6, th.imag(Sm1))
    plt.plot(t * 1e6, th.abs(Sm1))
    plt.grid()
    plt.legend(['Imaginary part', 'Amplitude'])
    plt.title('Convolution matched filter')
    plt.xlabel(r'Time/$\mu s$')
    plt.ylabel('Amplitude')
    plt.subplot(223)
    plt.plot(f, th.abs(Ym1))
    plt.grid()
    plt.subplot(224)
    plt.plot(f, th.angle(Ym1))
    plt.grid()

    plt.figure(2)
    plt.subplot(221)
    plt.plot(t * 1e6, th.real(Sm2))
    plt.plot(t * 1e6, th.abs(Sm2))
    plt.grid()
    plt.legend(['Real part', 'Amplitude'])
    plt.title('Correlation matched filter')
    plt.xlabel(r'Time/$\mu s$')
    plt.ylabel('Amplitude')
    plt.subplot(222)
    plt.plot(t * 1e6, th.imag(Sm2))
    plt.plot(t * 1e6, th.abs(Sm2))
    plt.grid()
    plt.legend(['Imaginary part', 'Amplitude'])
    plt.title('Correlation matched filter')
    plt.xlabel(r'Time/$\mu s$')
    plt.ylabel('Amplitude')
    plt.subplot(223)
    plt.plot(f, th.abs(Ym2))
    plt.grid()
    plt.subplot(224)
    plt.plot(f, th.angle(Ym2))
    plt.grid()
    plt.show()
