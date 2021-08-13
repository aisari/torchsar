#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-18 11:06:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function
import torch as th
from torchsar.utils.const import *


def dre_geo(Wl, Vr, R, Ar=0.):
    """doppler rate estimation based on geometry

    doppler rate estimation based on geometry

    Parameters
    ----------
    Wl : float
        Wave length.
    Vr : float, list or array
        Equivalent velocity of radar.
    R : float, list or array
        Minimum slant range from radar to target.
    Ar : float, list or array
        Equivalent squint angle.
    """

    raise TypeError('Not opened yet!')
