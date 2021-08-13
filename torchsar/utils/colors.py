#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


from __future__ import division, print_function, absolute_import

import torch as th


def rgb2gray(rgb, fmt='chnllast'):
    r"""Converts RGB image to GRAY image

    Converts RGB image to GRAY image according to

    .. math::
       G = 0.2989R + 0.5870G + 0.1140B

    see matlab's ``rgb2gray`` function for details.

    Parameters
    ----------
    rgb : Tensor
        Original RGB tensor.
    fmt : str, optional
        Specifies the position of channels in :attr:`rgb` tensor, surpported are:
        - ``'chnllast'`` (default)
        - ``'chnlfirst'``
    """

    dtype = rgb.dtype

    if rgb.dim() < 3:
        return rgb

    if fmt in ['chnllast', 'ChnlLast']:
        return (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]).to(dtype)
    if fmt in ['chnlfirst', 'ChnlFirst']:
        return (0.2989 * rgb[0, ...] + 0.5870 * rgb[1, ...] + 0.1140 * rgb[2, ...]).to(dtype)


if __name__ == '__main__':

    A = th.randn(5, 4, 3)
    print(A)
    print(rgb2gray(A, fmt='chnlfirst'))

    A = th.randn(3, 5, 4)
    print(A)
    print(rgb2gray(A, fmt='chnllast'))
