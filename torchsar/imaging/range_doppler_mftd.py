#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-18 11:06:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th
import numpy as np
import torchsar as ts
import matplotlib.pyplot as plt


def rda_mftd(Sr, pdict, mfmod='fftconv', iqc=False, rcmc=False, dcem=None, win=None, afa=None, ftshift=False, isplot=False, islog=True):
    r"""Range Doppler Algorithm with time domain matched filters

    Args:
        Sr (Tensor): The SAR complex echo data representated in real format :math:`{\bm S}_r \in \mathbb{R}^{N_a×N_r×2}`.
        pdict (dict): Parameters used in RDA, which are show as follows:
            ``{'Vr', 'R0', 'La', 'Fc', 'Tp', 'Kr',  Fsr', 'Fsa'}``
        mfmod (str, optional): Matched filter mode, supported are:

            - ``'corr1'`` : 1d correlation filter and use standard correlation
            - ``'conv1'`` : 1d convolution filter and use standard convolution
            - ``'fftcorr1'`` : 1d correlation filter and use fft for operating correlation
            - ``'fftconv1'`` : 1d convolution filter and use fft for operating convolution (default)

        iqc (bool, optional): Whether do IQ data correction, see :func:`iq_correct`:

            - I/Q bias removal
            - I/Q gain imbalance correction
            - I/Q non-orthogonality correction

        rcmc (bool, int, optional): Range Migration Correction: integer-->kernel size, ``False``-->no rcmc (default: {8})
        dcem (str, optional): Dopplor centroid frequency estimation method (the default is None, does not estimate)
            - ``'abdce_wda'`` :
        win (list, tuple or None, optional): the window function for matched filter of azimuth and range. If None, no window is added (default), e.g. ['kaiser 12', 'hanning'], this will add kaiser window and hanning window in azimuth and range respectively.
        afa (str, optional): Dopplor rate estimation (autofocus) method (the default is None, does not do autofocusing)
        ftshift (bool, optional): Whether to shift zeros frequency to center when use fft, ifft, fftfreq (the default is ``False``)
        isplot (bool, optional): Plot part of processing result, such as DCE result (default: ``False``)
        islog (bool, optional): Display processing info (default: ``True``)

    Returns:
        Tensor: Focused complex image :math:`{\bm S}_i \in \mathbb{C}^{N_a×N_r}`.
    """

    raise TypeError('Not opened yet!')
