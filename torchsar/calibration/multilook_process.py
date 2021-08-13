#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-05 11:06:13
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th


def multilook_spatial(Sslc, nlooks):
    r"""spatial multilook processing

    spatial averaging in azimuth or range direction.

    Args:
        Sslc (Tensor): Processed single look complex (or intensity) sar data tensor with size :math:`N_a×N_r`.
        nlooks (tuple or list): The number of looks in azimuth and range direction, [na, nr] or (na, nr).

    Returns:
        Smlc (Tensor): Processed multi-look complex tensor.

    """

    raise TypeError('Not opened yet!')


if __name__ == '__main__':

    Na, Nr = (1025, 256)
    real = th.randn(Na, Nr)
    imag = th.randn(Na, Nr)
    print(real.shape, imag.shape)
    Sslc = real + 1j * imag
    Smlc = multilook_spatial(Sslc, nlooks=(4, 1))

    print(Smlc.shape)
