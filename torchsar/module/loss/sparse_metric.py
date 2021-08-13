#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


class LogSparseLoss(th.nn.Module):
    r"""Log Sparse loss



    """

    def __init__(self, p=1., axis=None, caxis=None, reduction='mean'):
        super(LogSparseLoss, self).__init__()
        self.p = p
        self.axis = axis
        self.caxis = caxis
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):
            X = (X * X.conj()).real
        elif (self.caxis is None) or X.shape[-1] == 2:
            X = th.sum(X.pow(2), axis=-1, keepdims=True)
        else:
            X = th.sum(X.pow(2), axis=self.caxis, keepdims=True)

        if self.axis is None:
            D = X.dim()
            axis = list(range(1, D)) if D > 2 else list(range(0, D))
        else:
            axis = self.axis
        S = th.sum(th.log2(1 + X / self.p), axis)
        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


class FourierDomainLogSparseLoss(th.nn.Module):
    r"""FourierDomainLogSparseLoss



    """

    def __init__(self, p=1, axis=(-2, -1), caxis=None, reduction='mean'):
        super(FourierDomainLogSparseLoss, self).__init__()
        self.p = p
        self.axis = [axis] if type(axis) is int else axis
        self.reduction = reduction
        self.caxis = caxis

    def forward(self, X):
        D = X.dim()
        if self.axis is None:
            axis = list(range(1, D)) if D > 2 else list(range(0, D))
        else:
            axis = self.axis
        axis = [a + D if a < 0 else a for a in axis]

        if th.is_complex(X):
            caxis = None
        else:
            caxis = self.caxis + D if self.caxis < 0 else self.caxis
            if caxis != D - 1:
                newshape = list(range(0, caxis)) + list(range(caxis + 1, D)) + [caxis]
                X = X.permute(newshape)
                axis = [a if a < caxis else a - 1 for a in axis]

            X = X[..., 0] + 1j * X[..., 1]

        for a in axis:
            X = th.fft.fft(X, n=None, dim=a)

        X = X.abs()

        S = th.sum(th.log2(1 + X / self.p), axis)
        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


if __name__ == '__main__':

    p = 1
    p = 2
    p = 0.5
    X = th.randn(1, 3, 4, 2)
    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]

    sparse_func = LogSparseLoss(p=p)
    sparse_func = LogSparseLoss(p=p, axis=None, caxis=-1)
    sparse_func1 = LogSparseLoss(p=p, axis=None, caxis=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)

    sparse_func = FourierDomainLogSparseLoss(p=p)
    sparse_func = FourierDomainLogSparseLoss(p=p, axis=(1, 2), caxis=-1)
    sparse_func1 = FourierDomainLogSparseLoss(p=p, axis=(2, 3), caxis=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)
