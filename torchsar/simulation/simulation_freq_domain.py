#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import math
import torch as th
import torchsar as ts
import matplotlib.pyplot as plt


def dsm2sar_fd(g, pdict, mode='StationaryTarget2D', ftshift=True, device='cpu'):
    """Frequency-domain simulation

    SAR raw data simulation by frequency-domain method.

    Parameters
    ----------
    g : tensor
        The digital scene matrix.
    pdict : dict
        The SAR platform parameters.
    mode : str, optional
        Simulation mode.
    ftshift : bool, optional
        Shift zero-frequency to center?
    device : str, optional
        Specifies which device to be used.

    Returns
    -------
    tensor
        Simulated SAR raw data.
    """
    raise TypeError('Not opened yet!')
