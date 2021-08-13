#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


class FrobeniusLoss(th.nn.Module):
    r"""Frobenius norm loss

        .. math::
           \|\bm X\|_p^p = (\sum{x^p})^{1/p}

    """

    def __init__(self, p=2, axis=None, caxis=None, reduction='mean'):
        super(FrobeniusLoss, self).__init__()
        self.p = p
        self.axis = axis
        self.caxis = caxis
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):
            X = X.abs()
        elif (self.caxis is None) or X.shape[-1] == 2:
            X = X.pow(2).sum(axis=-1, keepdims=True).sqrt()
        else:
            X = X.pow(2).sum(axis=self.caxis, keepdims=True).sqrt()

        if X.dtype is not th.float32 or th.double:
            X = X.to(th.float32)

        if self.axis is None:
            D = X.dim()
            axis = list(range(1, D)) if D > 2 else list(range(0, D))
        else:
            axis = self.axis

        X = th.mean(X.pow(self.p), axis=axis).pow(1. / self.p)

        if self.reduction == 'mean':
            F = th.mean(X)
        if self.reduction == 'sum':
            F = th.sum(X)

        return F


class LogFrobeniusLoss(th.nn.Module):
    r"""LogFrobenius Loss

        .. math::
           {\log}{\||\bm X|\|_p^p} = \log {\left((\sum{|x|^p})^{1/p}\right)}

    """

    def __init__(self, p=2, axis=None, caxis=-1, reduction='mean'):
        super(LogFrobeniusLoss, self).__init__()
        self.p = p
        self.axis = axis
        self.caxis = caxis
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):
            X = X.abs()
        elif (self.caxis is None) or X.shape[-1] == 2:
            X = X.pow(2).sum(axis=-1, keepdims=True).sqrt()
        else:
            X = X.pow(2).sum(axis=self.caxis, keepdims=True).sqrt()

        if X.dtype is not th.float32 or th.double:
            X = X.to(th.float32)

        if self.axis is None:
            D = X.dim()
            axis = list(range(1, D)) if D > 2 else list(range(0, D))
        else:
            axis = self.axis

        X = th.mean(X.pow(self.p), axis=axis).pow(1. / self.p)

        if self.reduction == 'mean':
            F = th.mean(X)
        if self.reduction == 'sum':
            F = th.sum(X)

        return th.log(F)


if __name__ == '__main__':

    f_func = FrobeniusLoss(reduction='mean')
    X = th.randn(1, 3, 4, 2)
    V = f_func(X)
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = f_func(X)
    print(V)
