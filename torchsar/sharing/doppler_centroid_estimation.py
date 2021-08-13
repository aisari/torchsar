#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-02-18 11:06:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th
import numpy as np
from torchsar.utils.const import *
from torchsar.dsp.ffts import fft, ifft, fftfreq
from torchsar.dsp.polynomialfit import polyfit, polyval
import matplotlib.pyplot as plt


def bdce_sf(Sr, Fsa, Fsr, rdflag=False, Nroff=0, isplot=False):
    r"""Baseband doppler centroid estimation by spectrum fitting

    Baseband doppler centroid estimation by spectrum fitting.

    Parameters
    ----------
    Sr : numpy array
        SAR signal :math:`N_a×N_r` in range-doppler domain (range frequency domain).
    Fsa : float
        Sampling rate in azimuth
    rdflag : bool
        Specifies whether the input SAR signal is in range-doppler domain. If not, :func:`dce_sf` excutes FFT in range direction.
    isplot : bool
        Whether to plot the estimated results.(Default: False)
    """
    raise TypeError('Not opened yet!')


def bdce_api(Sr, Fsa, isplot=False):
    r"""Baseband doppler centroid estimation by average phase increment

    Baseband doppler centroid estimation by average phase increment.

    Parameters
    ----------
    Sr : numpy array
        SAR raw data or range compressed data
    Fsa : float
        Sampling rate in azimuth
    isplot : bool
        Whether to plot the estimated results.(Default: False)
    """

    raise TypeError('Not opened yet!')


def bdce_madsen(Sr, Fsa, isplot=False):
    r"""Baseband doppler centroid estimation by madsen

    Baseband doppler centroid estimation bymadsen.

    Parameters
    ----------
    Sr : numpy array
        SAR raw data or range compressed data
    Fsa : float
        Sampling rate in azimuth
    isplot : bool
        Whether to plot the estimated results.(Default: False)
    """

    raise TypeError('Not opened yet!')


def abdce_wda_ori(Sr, Fsa, Fsr, Fc, rate=0.9, isplot=False, islog=False):
    r"""Absolute and baseband doppler centroid estimation by wavelength diversity algorithm

    Absolute and baseband doppler centroid estimation by Wavelength Diversity Algorithm (WDA).

    <<合成孔径雷达成像_算法与实现>> p350.

    Parameters
    ----------
    Sr : numpy array
        SAR signal :math:`N_a×N_r` in range frequency domain.
    Fsa : float
        Sampling rate in azimuth.
    Fsr : float
        Sampling rate in range.
    """
    raise TypeError('Not opened yet!')


def abdce_wda_opt(Sr, Fsr, Fsa, Fc, ncpb=None, tr=None, isfftr=False, isplot=False, islog=False):
    """Absolute and baseband doppler centroid estimation by wavelength diversity algorithm

    Absolute and baseband doppler centroid estimation by Wavelength Diversity Algorithm (WDA).

    <<合成孔径雷达成像_算法与实现>> p350.

    Parameters
    ----------
    Sr : 2d-tensor
        SAR signal :math:`N_a×N_r` in range frequency domain.
    Fsr : float
        Sampling rate in range, unit Hz.
    Fsa : float
        Sampling rate in azimuth, unit Hz.
    Fc : float
        Carrier frequency, unit Hz.
    ncpb : tuple or list, optional
        Number of cells per block, so we have blocks (int(Na/ncpb[0])) × (int(Nr/ncpb[1]))
        (the default is [Na, Nr], which means all).
    tr : 1d-tensor, optional
        Time in range (the default is None, which linspace(0, Nr, Nr)).
    isplot : bool, optional
        Whether to plot the estimation results (the default is False).
    isfftr : bool, optional
        Whether to do FFT in range (the default is False).

    Returns
    -------
    fadc : 2d-tensor
        Absolute doppler centroid frequency, which has the size specified by :attr:`ncpb`.
    fbdc : 2d-tensor
        Baseband doppler centroid frequency, which has the size specified by :attr:`ncpb`.
    Ma : 2d-tensor
        Doppler ambiguity number, which has the size specified by :attr:`ncpb`.
    """
    raise TypeError('Not opened yet!')


def fullfadc(fdc, shape):
    nblks = fdc.shape
    Na, Nr = shape

    NBa = nblks[0]
    NBr = nblks[1]
    Na1b = int(np.uint(Na / NBa))
    Nr1b = int(np.uint(Nr / NBr))

    fc = th.zeros((Na, Nr))

    for a in range(NBa):
        for r in range(NBr):
            fc[int(a * Na1b):min(int((a + 1) * Na1b), Na), int(r * Nr1b):min(int((r + 1) * Nr1b), Nr)] = fdc[a, r]
    return fc


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torchsar

    datafile = '/mnt/d/DataSets/sar/ALOSPALSAR/mat/ALPSRP050500980-L1.0/ALOS_PALSAR_RAW=IMG-HH-ALPSRP050500980-H1(sl=1el=35345).mat'
    # datafile = '/mnt/d/DataSets/sar/ERS/mat/ERS2_SAR_RAW=E2_02558_STD_L0_F327(sl=1el=28682).mat'

    sardata, sarplat = torchsar.sarread(datafile)

    Fsa = sarplat.params['Fsa']
    Fsr = sarplat.params['Fsr']
    fr = sarplat.params['fr']
    Kr = sarplat.params['Kr']
    Fc = sarplat.sensor['Fc']
    Sr = sardata.rawdata[:, :, 0] + 1j * sardata.rawdata[:, :, 1]

    Sr = Sr[0:1024, 0:2048]
    # Sr = Sr[1024:4096, 0:2048]
    fr = th.linspace(-Fsr / 2.0, Fsr / 2.0, Sr.shape[1])

    # Sr = fftshift(fft(fftshift(Sr, axes=1), axis=1), axes=1)

    # Sr = torchsar.range_matched_filtering(Sr, fr, Kr)

    # Sr = ifftshift(ifft(ifftshift(Sr, axes=1), axis=1), axes=1)

    # aa = torchsar.doppler_center_estimation(Sr, Fsa)
    # print(aa)

    # Sr = Sr[512:-512, 512:-512]
    # Sr = Sr[:, 512:512+1024]
    # Sr = Sr[0:512, 0:512]

    # Sr = fftshift(fft(fftshift(Sr, axes=1), axis=1), axes=1)
    # Sr = fftshift(fft(Sr, axis=1), axes=1)
    # Sr = fft(Sr, axis=1)

    print(Sr.shape)

    accc(Sr, isplot=True)

    _, dc_coef = bdce_sf(Sr, Fsa, rdflag=False, isplot=True)
    print(dc_coef)
    bdce_api(Sr, Fsa, isplot=True)

    fadc = abdce_wda_ori(Sr, Fsa, Fsr, Fc)

    print(fadc)
