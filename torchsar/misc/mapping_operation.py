#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 22:29:14
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

from torchsar.base.mathops import nextpow2
from torchsar.misc.transform import scale
from torchsar.utils.const import EPS
from torchsar.utils.convert import str2num
import torch as th


def mapping(X, drange=(0., 255.), mode='amplitude', method='2Sigma', odtype='auto'):
    r"""convert to image

    Convert data to image data :math:`\bm X` with dynamic range :math:`d=[{\rm min}({\bm X}), {\rm max}({\bm X})]`.

    Args:
        X (Tensor): data to be converted
        drange (tuple, optional): dynamic range (the default is (0., 255.))
        mode (str, optional): data mode in :attr:`X`, ``'amplitude'`` (default) or ``'power'``.
        method (str, optional): converting method, surpported values are ``'1Sigma'``, ``'2Sigma'``, ``'3Sigma'``
            (the default is '2Sigma', which means two-sigma mapping)
        odtype (str, None... }, optional): output data type, supportted are ``'auto'`` (auto infer, default),
            or torch tensor's dtype string.
            If the type of :attr:`odtype` is not string, the output data type is ``'th.float32'``.

    Returns:
        Y (Tensor): converted image data

    """

    if method is None:
        return X

    X = X.float()
    xmin, xmax = X.min(), X.max()
    dmin, dmax = drange[0:2]

    if method[-5:] in ['Sigma', 'sigma', 'SIGMA']:
        nsigma = str2num(method, float)[0]

        if mode in ['Amplitude', 'amplitude', 'AMPLITUDE']:
            xvsv = X.std()
        if mode in ['Power', 'power', 'POWER']:
            xvsv = X.var()

        xmean = X.mean()
        diff_min = xmean - nsigma * xvsv
        diff_max = xmean + nsigma * xvsv

        ymin, ymax = diff_min, diff_max

        if diff_min < xmin:
            ymin = xmin
        if diff_max > xmax:
            ymax = xmax

        slope = dmax / (ymax - ymin + EPS)
        # offset = -slope * ymin
        offset = -slope * ymin + dmin

        X = slope * X + offset
        X[X < dmin] = dmin
        X[X > dmax] = dmax

    if method in ['Log', 'log', 'LOG']:
        X = X / th.max(X)
        X = 20.0 * th.log10(X)
        X[X < drange[-1]] = drange[-1]
        X = scale(X, drange[0:2], [drange[-1], 0])

    if odtype in ['auto', 'AUTO']:
        if dmin >= 0:
            odtype = 'th.uint'
        else:
            odtype = 'th.int'
        odtype = odtype + str(nextpow2(drange[1] - drange[0]))

    if type(odtype) is str:
        X = X.to(eval(odtype))
    return X


if __name__ == '__main__':

    X = th.randn(3, 4)
    X = X.abs()

    print(X)

    X = mapping(X)

    print(X)
