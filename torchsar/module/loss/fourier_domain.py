#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


class FourierDomainAmplitudeLoss(th.nn.Module):
    r"""Fourier Domain Amplitude Loss



    """

    def __init__(self, mode='mse', axis=(-2, -1), caxis=None, norm=None, reduction='mean'):
        super(FourierDomainAmplitudeLoss, self).__init__()
        self.mode = mode
        self.axis = [axis] if type(axis) is int else axis
        self.norm = 'max' if norm is True else norm
        self.reduction = reduction
        self.caxis = caxis

        if self.mode in ['mse', 'MSE', 'Mse']:
            self.lossfn = th.nn.MSELoss(reduction=self.reduction)
        if self.mode in ['mae', 'MAE', 'Mae']:
            self.lossfn = th.nn.L1Loss(reduction=self.reduction)

    def forward(self, P, G):
        D = P.dim()
        if self.axis is None:
            axis = list(range(1, D)) if D > 2 else list(range(0, D))
        else:
            axis = self.axis
        axis = [a + D if a < 0 else a for a in axis]
        caxis = self.caxis
        if th.is_complex(P):
            caxis = None
        if caxis is not None:
            caxis = self.caxis + D if self.caxis < 0 else self.caxis
            if caxis != D - 1:
                newshape = list(range(0, caxis)) + list(range(caxis + 1, D)) + [caxis]
                P = P.permute(newshape)
                G = G.permute(newshape)
                axis = [a if a < caxis else a - 1 for a in axis]

            P = P[..., 0] + 1j * P[..., 1]
            G = G[..., 0] + 1j * G[..., 1]

        if self.norm in ['max', 'MAX', 'Max']:
            maxv = G.abs().max() + 1e-16
            P = P / maxv
            G = G / maxv

        for a in axis:
            P = th.fft.fft(P, n=None, dim=a)
            G = th.fft.fft(G, n=None, dim=a)

        P, G = P.abs(), G.abs()

        return self.lossfn(P, G)


class FourierDomainPhaseLoss(th.nn.Module):
    r"""Fourier loss



    """

    def __init__(self, mode='mse', axis=(-2, -1), caxis=None, norm=None, reduction='mean'):
        super(FourierDomainPhaseLoss, self).__init__()
        self.mode = mode
        self.axis = [axis] if type(axis) is int else axis
        self.caxis = caxis
        self.norm = 'max' if norm is True else norm
        self.reduction = reduction

        if self.mode in ['mse', 'MSE', 'Mse']:
            self.lossfn = th.nn.MSELoss(reduction=self.reduction)
        if self.mode in ['mae', 'MAE', 'Mae']:
            self.lossfn = th.nn.L1Loss(reduction=self.reduction)

    def forward(self, P, G):
        D = P.dim()
        if self.axis is None:
            axis = list(range(1, D)) if D > 2 else list(range(0, D))
        else:
            axis = self.axis
        axis = [a + D if a < 0 else a for a in axis]
        caxis = self.caxis
        if th.is_complex(P):
            caxis = None
        if caxis is not None:
            caxis = self.caxis + D if self.caxis < 0 else self.caxis
            if caxis != D - 1:
                newshape = list(range(0, caxis)) + list(range(caxis + 1, D)) + [caxis]
                P = P.permute(newshape)
                G = G.permute(newshape)
                axis = [a if a < caxis else a - 1 for a in axis]

            P = P[..., 0] + 1j * P[..., 1]
            G = G[..., 0] + 1j * G[..., 1]

        if self.norm in ['max', 'MAX', 'Max']:
            maxv = G.abs().max() + 1e-16
            P = P / maxv
            G = G / maxv

        for a in axis:
            P = th.fft.fft(P, n=None, dim=a)
            G = th.fft.fft(G, n=None, dim=a)

        P, G = P.angle(), G.angle()

        return self.lossfn(P, G)


class FourierDomainLoss(th.nn.Module):
    r"""Fourier loss



    """

    def __init__(self, mode='mse', axis=(-2, -1), caxis=None, norm=None, reduction='mean'):
        super(FourierDomainLoss, self).__init__()
        self.mode = mode
        self.axis = [axis] if type(axis) is int else axis
        self.norm = 'max' if norm is True else norm
        self.caxis = caxis
        self.reduction = reduction

    def forward(self, P, G):

        D = P.dim()
        if self.axis is None:
            axis = list(range(1, D)) if D > 2 else list(range(0, D))
        else:
            axis = self.axis
        axis = [a + D if a < 0 else a for a in axis]
        caxis = self.caxis
        if th.is_complex(P):
            caxis = None
        if caxis is not None:
            caxis = self.caxis + D if self.caxis < 0 else self.caxis
            if caxis != D - 1:
                newshape = list(range(0, caxis)) + list(range(caxis + 1, D)) + [caxis]
                P = P.permute(newshape)
                G = G.permute(newshape)
                axis = [a if a < caxis else a - 1 for a in axis]

            P = P[..., 0] + 1j * P[..., 1]
            G = G[..., 0] + 1j * G[..., 1]

        if self.norm in ['max', 'MAX', 'Max']:
            maxv = G.abs().max() + 1e-16
            P = P / maxv
            G = G / maxv

        for a in axis:
            P = th.fft.fft(P, n=None, dim=a)
            G = th.fft.fft(G, n=None, dim=a)

        if self.mode in ['mse', 'MSE', 'Mse']:
            L = ((P - G) * ((P - G).conj())).real().mean(axis=axis)
        if self.mode in ['mae', 'MAE', 'Mae']:
            L = (P - G).abs().mean(axis=axis)

        if self.reduction == 'mean':
            return th.mean(L)
        if self.reduction == 'sum':
            return th.sum(L)


class FourierDomainNormLoss(th.nn.Module):
    r"""FourierDomainNormLoss

    .. math::
        C = \frac{{\rm E}(|I|^2)}{[E(|I|)]^2}

    see Fast Fourier domain optimization using hybrid

    """

    def __init__(self, reduction='mean', p=1.5):
        super(FourierDomainNormLoss, self).__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, X, w=None):
        r"""[summary]

        [description]

        Parameters
        ----------
        X : {[type]}
            After fft in azimuth
        w : {[type]}, optional
            [description] (the default is None, which [default_description])

        Returns
        -------
        [type]
            [description]
        """

        if th.is_complex(X):
            X = X.abs()
        elif X.shape[-1] == 2:
            X = th.view_as_complex(X)
            X = X.abs()

        if w is None:
            wshape = [1] * (X.dim())
            wshape[-2] = X.size(-2)
            w = th.ones(wshape, device=X.device, dtype=X.dtype)
        fv = th.sum((th.sum(w * X, axis=-2)).pow(self.p), axis=-1)

        if self.reduction == 'mean':
            C = th.mean(fv)
        if self.reduction == 'sum':
            C = th.sum(fv)
        return C


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(1, 3, 4, 2) * 10000
    Y = th.randn(1, 3, 4, 2) * 10000

    fdal_func = FourierDomainAmplitudeLoss(mode='mse', axis=(1, 2), caxis=-1, norm=None, reduction='mean')
    S = fdal_func(X, X)
    print(S)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    print(X.abs())
    Y = X.clone()
    axis = (1, 2)
    fdal_func = FourierDomainAmplitudeLoss(mode='mse', axis=(1, 2), norm='max', reduction='mean')
    for a in axis:
        Y = th.fft.fft(Y, dim=a)
    Y = Y * th.exp(1j * th.rand(X.shape))
    for a in axis:
        Y = th.fft.ifft(Y, dim=a)
    print(Y.abs())

    print(X.shape, Y.shape, "++++")
    S = fdal_func(X, Y)
    print(S)

    print(Y.abs())

    Z = Y * th.exp(1j * th.rand(X.shape))

    print(Z.abs())

    th.manual_seed(2020)
    X = th.randn(1, 2, 3, 4) * 10000
    Y = th.randn(1, 2, 3, 4) * 10000
    fdal_func = FourierDomainAmplitudeLoss(mode='mse', axis=(2, 3), caxis=1, norm='max', reduction='mean')
    S = fdal_func(X, Y)
    print(S)
