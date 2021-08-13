#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


class CMSELoss(th.nn.Module):
    r"""Complex mean squared error loss


    """

    def __init__(self, caxis=-1, norm='max', reduction='mean'):
        super(CMSELoss, self).__init__()
        self.norm = norm
        self.caxis = caxis
        self.reduction = reduction

    def forward(self, P, G):
        D = P.dim()
        caxis = self.caxis
        if th.is_complex(P):
            caxis = None
        if caxis is not None:
            caxis = self.caxis + D if self.caxis < 0 else self.caxis
            if caxis != D - 1:
                newshape = list(range(0, caxis)) + list(range(caxis + 1, D)) + [caxis]
                P = P.permute(newshape)
                G = G.permute(newshape)

            if P.shape[-1] == 2:
                P = P[..., 0] + 1j * P[..., 1]
            if G.shape[-1] == 2:
                G = G[..., 0] + 1j * G[..., 1]

        axis = list(range(1, P.dim()))

        if self.norm in ['max', 'MAX', 'Max']:
            maxv = G.abs().max() + 1e-16
            P = P / maxv
            G = G / maxv

        F = th.mean((P - G).abs().pow(2), axis=axis)

        if self.reduction == 'mean':
            F = th.mean(F)
        if self.reduction == 'sum':
            F = th.sum(F)

        return F


class CMAELoss(th.nn.Module):
    r"""Complex mean absoluted error loss


    """

    def __init__(self, caxis=-1, norm='max', reduction='mean'):
        super(CMAELoss, self).__init__()
        self.reduction = reduction
        self.caxis = caxis
        self.norm = norm

    def forward(self, P, G):
        D = P.dim()
        caxis = self.caxis
        if th.is_complex(P):
            caxis = None
        if caxis is not None:
            caxis = self.caxis + D if self.caxis < 0 else self.caxis
            if caxis != D - 1:
                newshape = list(range(0, caxis)) + list(range(caxis + 1, D)) + [caxis]
                P = P.permute(newshape)
                G = G.permute(newshape)

            if P.shape[-1] == 2:
                P = P[..., 0] + 1j * P[..., 1]
            if G.shape[-1] == 2:
                G = G[..., 0] + 1j * G[..., 1]

        axis = list(range(1, P.dim()))

        if self.norm in ['max', 'MAX', 'Max']:
            maxv = G.abs().max() + 1e-16
            P = P / maxv
            G = G / maxv

        F = th.mean((P - G).abs(), axis=axis)

        if self.reduction == 'mean':
            F = th.mean(F)
        if self.reduction == 'sum':
            F = th.sum(F)

        return F


if __name__ == '__main__':

    X = th.randn(1, 3, 4, 2)
    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    Y = th.randn(1, 3, 4, 2)
    Y = Y[:, :, :, 0] + 1j * Y[:, :, :, 1]
    # Y = X

    errfn = CMAELoss()
    S = errfn(X, Y)
    print(S)

    errfn = CMAELoss(caxis=-1)
    Xr = th.view_as_real(X)
    Yr = th.view_as_real(Y)
    S = errfn(Xr, Yr)
    print(S)

    errfn = CMAELoss(caxis=1)
    Xr = Xr.permute(0, 3, 1, 2)
    Yr = Yr.permute(0, 3, 1, 2)
    S = errfn(Xr, Yr)
    print(S)

    errfn = CMSELoss()
    S = errfn(X, Y)
    print(S)

    errfn = CMSELoss(caxis=-1)
    Xr = th.view_as_real(X)
    Yr = th.view_as_real(Y)
    S = errfn(Xr, Yr)
    print(S)

    errfn = CMSELoss(caxis=1)
    Xr = Xr.permute(0, 3, 1, 2)
    Yr = Yr.permute(0, 3, 1, 2)
    S = errfn(Xr, Yr)
    print(S)
