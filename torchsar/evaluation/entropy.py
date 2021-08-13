#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchsar.utils.const import EPS


def entropy(X, mode='shannon', reduction='mean'):
    r"""compute the entropy of the inputs

    [description]

    Parameters
    ----------
    X : Tensor
        The complex or real inputs, for complex inputs, both complex and real representations are surpported.
    mode : str, optional
        The entropy mode: ``'shannon'`` or ``'natural'`` (the default is 'shannon')
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : Tensor
        The entropy of the inputs.
    """

    if mode in ['Shannon', 'shannon', 'SHANNON']:
        logfunc = th.log2
    if mode in ['Natural', 'natural', 'NATURAL']:
        logfunc = th.log

    if th.is_complex(X):
        X = (X * X.conj()).real
    elif X.size(-1) == 2:
        X = th.sum(X.pow(2), axis=-1)

    if X.dim() == 2:
        axis = (0, 1)
    if X.dim() == 3:
        axis = (1, 2)
    if X.dim() == 4:
        axis = (1, 2, 3)

    P = th.sum(X, axis, keepdims=True)
    p = X / (P + EPS)
    S = -th.sum(p * logfunc(p + EPS), axis)
    if reduction == 'mean':
        S = th.mean(S)
    if reduction == 'sum':
        S = th.sum(S)

    return S


if __name__ == '__main__':

    X = th.randn(1, 3, 4, 2)
    S = entropy(X, mode='shannon')
    print(S)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    S = entropy(X, mode='shannon')
    print(S)
