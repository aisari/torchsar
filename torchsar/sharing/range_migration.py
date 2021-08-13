#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-18 11:06:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th
import torchsar as ts


def rcmc_interp(Sr, tr, D):
    r"""range migration correction with linear interpolation

    Range migration correction with linear interpolation.

    Parameters
    ----------
    Sr : tensor
        SAR raw data :math:`N_a×N_r` in range dopplor domain
    tr : 1d-tensor
        time array :math:`N_r×1` in range
    D : 1d-tensor
        :math:`N_a×1` migration factor vector

    Returns
    -------
    Srrcmc
        data after range migration correction :math:`N_a×N_r`
    """

    raise TypeError('Not opened yet!')
