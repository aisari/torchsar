#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import math
import torch as th
import torchsar as ts


def tgs2sar_td(targets, pdict, mode='Target', device='cpu'):
    """Time-domain simulation

    SAR raw data simulation by time-domain method.

    Parameters
    ----------
    targets : tensor
        A 2d-tensor contains the information of targets,
        each row is a target [x, y, z, vx, vy, vz, a].
    pdict : dict
        The SAR platform parameters.
    mode : str, optional
        Simulation mode.
    device : str, optional
        Specifies which device to be used.

    Returns
    -------
    tensor
        Simulated SAR raw data tensor.
    """
    raise TypeError('Not opened yet!')
