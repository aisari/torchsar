#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th


def sparse_degree(x, t, mode='percentile'):
    x = x.abs()
    if mode in ['percentile', 'Percentile']:
        k = np.percentile(x, t)
        k = (x > k).sum().item()
    return k
