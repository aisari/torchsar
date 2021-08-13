#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-15 15:52:43
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import math
import torch as th
from torchsar.utils.const import *


def slantr2groundr(R, H, Ar, Xc):
    """slant range to ground range

    Convert slant range :math:`R` to ground range :math:`X`.

    Parameters
    ----------
    R : 1d-tensor
        slant range array
    H : float
        sarplat height
    Ar : float
        squint angle (unit, rad) in line geometry

    Returns
    -------
    X : 1d-tensor
        ground range

    """
    return th.sqrt(th.abs((R * math.cos(Ar))**2 - H * H) + EPS) - Xc
    # return th.sqrt(th.abs((R * math.cos(Ar))**2 - H * H) + EPS)


def slantt2groundr(tr, H, Ar):
    """slant time to ground range

    Convert slant range time :math:`t_r` to ground range :math:`X`.

    Parameters
    ----------
    tr : 1d-tensor
        range time array
    H : float
        sarplat height
    Ar : float
        squint angle (unit, rad) in line geometry

    Returns
    -------
    X : 1d-tensor
        ground range

    """

    return th.sqrt(th.abs(((tr * C / 2.) * math.cos(Ar))**2 - H * H) + EPS)


def groundr2slantr(X, H, Ar, Xc):
    """ground range to slant range

    Convert ground range :math:`R` to slant range :math:`X`.

    Parameters
    ----------
    X : 1d-tensor
        ground range array
    H : float
        sarplat height
    Ar : float
        squint angle (unit, rad) in line geometry

    Returns
    -------
    R : 1d-tensor
        slant range

    """

    return th.sqrt((X + Xc)**2 + H * H + EPS) / (math.cos(Ar) + EPS)
    # return th.sqrt((X)**2 + H * H + EPS) / (math.cos(Ar) + EPS)


def groundr2slantt(X, H, Ar):
    """ground range to slant time

    Convert ground range :math:`X` to slant time :math:`t_r`.

    Parameters
    ----------
    X : 1d-tensor
        ground range
    H : float
        sarplat height
    Ar : float
        squint angle (unit, rad) in line geometry

    Returns
    -------
    tr : 1d-tensor
        range time array

    """

    return 2. * th.sqrt(X**2 + H * H + EPS) / (math.cos(Ar) + EPS) / C


def min_slant_range(Rnear, Fsr, Noff):
    """minimum slant range from radar to target

    Compute the minimum slant range from radar to target.

    Parameters
    ----------
    Rnear : float
        The nearest range (start sampling) from radar to the target.
    Fsr : float
        Sampling rate in range direction
    Noff : 1d-tensor
        Offset from the start distance (start sampling) cell.

    Returns
    -------
    r : 1d-tensor
        Minimum slant range of each range cell.
    """
    r = Rnear + Noff * (C / (2. * Fsr))
    return r


def min_slant_range_with_migration(Rnear, Fsr, Noff, Wl, Vr, fdc):
    r = Rnear + Noff * (C / (2. * Fsr))
    return r / th.sqrt(1. - (Wl * fdc / (Vr + Vr))**2)
