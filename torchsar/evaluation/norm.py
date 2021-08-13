#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


def frobenius(X, p=2, reduction='mean'):
    r"""frobenius norm

        .. math::
           \|\bm X\|_p^p = (\sum{x^p})^{1/p}

    """

    if th.is_complex(X):
        X = ((X * X.conj()).real).sqrt()
    elif X.size(-1) == 2:
        X = X.pow(2).sum(axis=-1).sqrt()

    if X.dtype is not th.float32 or th.double:
        X = X.to(th.float32)

    D = X.dim()
    dim = list(range(1, D))
    X = th.mean(X.pow(p), axis=dim).pow(1. / p)

    if reduction == 'mean':
        F = th.mean(X)
    if reduction == 'sum':
        F = th.sum(X)

    return F


if __name__ == '__main__':

    X = th.randn(1, 3, 4, 2)
    V = frobenius(X, p=2, reduction='mean')
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = frobenius(X, p=2, reduction='mean')
    print(V)
