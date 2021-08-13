#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-05 11:06:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import math
import torch as th


def iq_correct(Sr):
    r"""IQ Correction performs the I Q data correction

    IQ Correction performs the I Q data correction

    - I/Q bias removal
    - I/Q gain imbalance correction
    - I/Q non-orthogonality correction

    see "Sentinel-1-Level-1-Detailed-Algorithm-Definition".

    Args:
        Sr (Tensor): SAR raw data matrix :math:`{\bm S}_r \in {\mathbb R}^{N_a×N_r×2}`

    Returns:
        Sr (Tensor): Corrected SAR raw data.
        Flag : dict

    """

    raise TypeError('Not opened yet!')


if __name__ == '__main__':

    Na, Nr = (4, 5)
    theta = th.rand(Na, Nr)

    Is = th.cos(theta)
    Qs = th.sin(theta)
    Sr = th.zeros((Na, Nr, 2))
    Sr[:, :, 0] = Is
    Sr[:, :, 1] = Qs

    print(Sr)
    Sr, Flag = iq_correct(Sr)
    print(Sr)
    print(Flag)
