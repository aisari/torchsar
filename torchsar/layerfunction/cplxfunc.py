#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 22:29:14
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

import torch as th
from torchsar.utils.const import EPS
from torch.nn.functional import relu


def csign(x, caxis=None):
    r"""The signum function like Matlab's sign

    .. math::
        {\rm csign}(x+jy) = \frac{x+jy}{\sqrt{x^2+y^2}}

    Parameters
    ----------
    x : tensor, int, float or complex
        The input
    caxis : int or None, optional
        Specifies the complex axis..

    Returns
    -------
    tensor
        The output.

    Raises
    ------
    TypeError
        :attr:`caxis` should be integer!
    """
    xtype = type(x)
    if (xtype is int) or (xtype is float) or (xtype is complex):
        return x / (abs(x) + EPS)
    if type(x) is not th.Tensor:
        x = th.tensor(x)
    if caxis is None:
        return x / (x.abs() + EPS)
        # return th.sgn(x)
    if type(caxis) is not int:
        raise TypeError('axis should be integer!')
    x = x / (x.pow(2).sum(caxis, keepdim=True).sqrt() + EPS)
    # x = x.transpose(caxis, -1)
    # x = th.view_as_complex(x)
    # x = th.sgn(x)
    # x = th.view_as_real(x)
    # x = x.transpose(caxis, -1)
    return x


def csoftshrink(x, alpha=0.5, caxis=None, inplace=False):
    r"""Complex soft shrink function

    Parameters
    ----------
    x : tensor
        The input.
    alpha : float, optional
        The threshhold.
    caxis : int or None, optional
        Specifies the complex axis.

    Returns
    -------
    tensor
        The output.

    Raises
    ------
    TypeError
        :attr:`caxis` should be integer!
    """
    if caxis is None:
        return csign(x, caxis=caxis) * relu(x.abs() - alpha, inplace=inplace)
    if type(caxis) is not int:
        raise TypeError('axis should be integer!')
    return csign(x, caxis=caxis) * relu(x.pow(2).sum(caxis, keepdim=True).sqrt() - alpha, inplace=inplace)

    # if caxis is None:
    #     x = th.sgn(x) * relu(x.abs() - alpha)
    #     return x
    # if type(caxis) is not int:
    #     raise TypeError('axis should be integer!')
    # x = x.transpose(caxis, -1)
    # x = th.view_as_complex(x)
    # x = th.sgn(x) * relu(x.abs() - alpha)
    # x = th.view_as_real(x)
    # x = x.transpose(caxis, -1))
    # return x


def softshrink(x, alpha=0.5, inplace=False):
    r"""Real soft shrink function

    Parameters
    ----------
    x : tensor
        The input.
    alpha : float, optional
        The threshhold.

    Returns
    -------
    tensor
        The output.
    """
    return th.sgn(x) * relu(x.abs() - alpha, inplace=inplace)


if __name__ == '__main__':

    x = 2 + 2j
    print(csign(x))

    x = [1 + 1j, 2 + 2j]
    print(csign(x))

    print(csoftshrink(th.tensor(x)))
    print(softshrink(th.tensor(x)))
