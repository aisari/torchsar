# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-23 19:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
import torch as th
from torchsar.base.arrayops import cut
from torchsar.utils.const import *


def mfpc_throwaway(X, No, Nh, axis=0, mffdmod='way1', ftshift=False):
    r"""Throwaway invalid pulse compressed data

    Throwaway invalid pulse compressed data

    Parameters
    ----------
    X : Tensor
        Data after pulse compression.
    No : int
        Output size.
    Nh : int
        Filter size.
    axis : int, optional
        Throwaway dimension. (the default is 0)
    mffdmod : str, optional
        Frequency filter mode. (the default is 'way1')
    ftshift : bool, optional
        Whether to shift frequency (the default is False)
    """

    raise TypeError('Not opened yet!')


if __name__ == '__main__':

    X = th.tensor(range(32))
    print(X, X.shape[0])

    No, Nh = (24, 7)
    X = mfpc_throwaway(X, No, Nh, axis=0, mffdmod='way1', ftshift=True)
    print(X)
