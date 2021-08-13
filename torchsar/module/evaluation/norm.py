#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


class Frobenius(th.nn.Module):
    r"""Frobenius

        .. math::
           \|\bm X\|_p^p = (\sum{x^p})^{1/p}

    """

    def __init__(self, p=2, reduction='mean'):
        super(Frobenius, self).__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, X):

        if th.is_complex(X):
            X = ((X * X.conj()).real).sqrt()
        elif X.size(-1) == 2:
            X = X.pow(2).sum(axis=-1).sqrt()

        if X.dtype is not th.float32 or th.double:
            X = X.to(th.float32)

        D = X.dim()
        dim = list(range(1, D))
        X = th.mean(X.pow(self.p), axis=dim).pow(1. / self.p)

        if self.reduction == 'mean':
            F = th.mean(X)
        if self.reduction == 'sum':
            F = th.sum(X)

        return F


class LogFrobenius(th.nn.Module):
    r"""LogFrobenius

        .. math::
           {\log}{\||\bm X|\|_p^p} = \log {\left((\sum{|x|^p})^{1/p}\right)}

    """

    def __init__(self, p=2, reduction='mean'):
        super(LogFrobenius, self).__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, X):

        if th.is_complex(X):
            X = ((X * X.conj()).real).sqrt()
        elif X.size(-1) == 2:
            X = X.pow(2).sum(axis=-1).sqrt()

        if X.dtype is not th.float32 or th.double:
            X = X.to(th.float32)

        D = X.dim()
        dim = list(range(1, D))
        X = th.mean(X.pow(self.p), axis=dim).pow(1. / self.p)

        if self.reduction == 'mean':
            F = th.mean(X)
        if self.reduction == 'sum':
            F = th.sum(X)

        return th.log(F)


if __name__ == '__main__':

    f_func = Frobenius(reduction='mean')
    X = th.randn(1, 3, 4, 2)
    V = f_func(X)
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = f_func(X)
    print(V)
