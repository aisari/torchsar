#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torchsar.utils.const import EPS


class EntropyLoss(th.nn.Module):
    r"""Entropy loss

    .. math::
        \mathrm{S}=-\sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \frac{|\mathbf{X}|_{i j}^{2}}{Z} \ln \frac{|\mathbf{X}|_{i j}^{2}}{Z}

    .. math::
        Z=\sum_{i=0}^{H-1} \sum_{j=0}^{W-1}|\mathbf{X}|_{i j}^{2}

    """

    def __init__(self, mode='shannon', axis=None, caxis=None, reduction='mean'):
        super(EntropyLoss, self).__init__()
        self.mode = mode
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

        P = th.sum(X, axis, keepdims=True)
        p = X / (P + EPS)
        if self.mode in ['Shannon', 'shannon', 'SHANNON']:
            S = -th.sum(p * th.log2(p + EPS), axis)
        if self.mode in ['Natural', 'natural', 'NATURAL']:
            S = -th.sum(p * th.log(p + EPS), axis)
        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


if __name__ == '__main__':

    ent_func = EntropyLoss('shannon')
    ent_func = EntropyLoss('natural')
    ent_func1 = EntropyLoss('natural', caxis=1)
    X = th.randn(1, 3, 4, 2)
    S = ent_func(X)
    print(S)

    Y = X.permute(0, 3, 1, 2)
    S = ent_func1(Y)
    print(S)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    S = ent_func(X)
    print(S)
