#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-23 19:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th


def window(n, wtype=None, isperiodic=True, dtype=None, device=None, requires_grad=False):
    r"""Generates window

    Parameters
    ----------
    n : int
        The length of the window.
    wtype : str or None, optional
        The window type:
        - ``'rectangle'`` for rectangle window
        - ``'bartlett'`` for bartlett window
        - ``'blackman'`` for blackman window
        - ``'hamming x y'`` for hamming window with :math:`\alpha=x, \beta=y`, default is 0.54, 0.46.
        - ``'hanning'`` for hanning window
        - ``'kaiser x'`` for kaiser window with :math:`\beta=x`, default is 12.
    isperiodic : bool, optional
        If True (default), returns a window to be used as periodic function.
        If False, return a symmetric window.
    dtype : None, optional
        The desired data type of returned tensor.
    device : None, optional
        The desired device of returned tensor.
    requires_grad : bool, optional
        If autograd should record operations on the returned tensor. Default: False.

    Returns
    -------
    tensor
        A 1-D tensor of size (n,) containing the window
    """

    raise TypeError('Not opened yet!')


def windowing(x, w, axis=None):
    """Performs windowing operation in the specified axis.

    Parameters
    ----------
    x : tensor
        The input tensor.
    w : tensor
        A 1-d window tensor.
    axis : int or None, optional
        The axis.

    Returns
    -------
    tensor
        The windowed data.

    """
    if axis is None:
        return x * w

    if type(axis) is not int:
        raise TypeError('The axis should be a integer!')

    d = x.dim()
    shape = [1] * d
    shape[axis] = len(w)

    w = w.view(shape)
    return x * w


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 512
    wtype = 'bartlett'
    wtype = 'blackman'
    wtype = 'hamming 0.54 0.46'
    wtype = 'hanning'
    wtype = 'kaiser 12'
    w = window(n, wtype=wtype)

    plt.figure()
    plt.grid()
    plt.plot(w)
    plt.show()
