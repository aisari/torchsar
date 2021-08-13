#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-18 11:06:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function
import numpy as np
from torchsar.utils.const import *


def azimuth_beamwidth(Wl, La):
    BWa = 0.886 * Wl / La
    return BWa


def azimuth_footprint(R, Wl, La):

    BWa = 0.886 * Wl / La
    return R * BWa


def compute_range_beamwidth2(Nr, Fsr, H, Aon):
    r"""computes beam angle in range direction


    Parameters
    ----------
    Nr : int
        Number of samples in range direction.
    Fsr : float
        Sampling rate in range direction.
    H : float
        Height of the platform.
    Aon : float
        The off-nadir angle (unit: rad).

    Returns
    -------
    float
        The beam angle (unit, rad) in range direction.

    """

    Rnear = H / (np.cos(Aon) + EPS)
    Rfar = Rnear + Nr * (C / (2. * Fsr))

    return np.arccos(H / Rfar) - abs(Aon)


def cr_footprint(Wl, H, La, Ad):
    r"""cross range (azimuth) foot print

    .. math::
       R_{CR} \approx \frac{\lambda}{L_a}\frac{H}{{\rm cos}\theta_d}

    Parameters
    ----------
    Wl : float
        wave length
    H : float
        height of SAR platform
    La : float
        length of antenna aperture (azimuth)
    Ad : float
        depression angle

    Returns
    -------
    float
        foot print size in azimuth
    """

    FPa = (Wl * H) / (La * np.cos(Ad))

    return FPa
