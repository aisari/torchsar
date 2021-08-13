#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import torch as th
import torchsar as ts


def __fdnorm(X, p):

    X = th.sum(X.abs(), axis=-1).sqrt()
    # if th.is_complex(X):  # N-Na-Nr
    #     X = X.abs()
    # elif X.size(-1) == 2:  # N-Na-Nr-2
    #     X = th.sum(X.pow(2), axis=-1).sqrt()
    print(th.sum(X), "]]]]")
    C = th.sum(th.sum(X, axis=-2).pow(p), axis=-1)
    print(C.shape, C)
    return th.mean(C)


def af_ffo_sm(x, p=2, niter=10, delta=None, toli=1e-2, tolo=1e-2, ftshift=True, islog=False):
    """Stage-by-Stage minimum entropy

    [1] Morrison Jr, Robert Lee; Autofocus, Entropy-based (2002): Entropy-based autofocus for synthetic aperture radar.

    Parameters
    ----------
    x : Tensor
       Corrupt complex SAR image. N-Na-Nr(complex) or N-Na-Nr-2(real)
    niter : int, optional
       The number of iteration (the default is 10)
    delta : {float or None}, optional
       The change step (the default is None (i.e. PI))
    toli : int, optional
       Tolerance error for inner loop (the default is 1e-2)
    tolo : int, optional
       Tolerance error for outer loop (the default is 1e-2)
    ftshift : bool, optional
       Shift the zero frequency to center? (the default is True)
    islog : bool, optional
       Print log information? (the default is False)
    """

    if delta is None:
        delta = ts.PI

    if th.is_complex(x):
        x = th.view_as_real(x)
        cplxflag = True
    elif x.size(-1) == 2:
        cplxflag = False
    else:
        raise TypeError('x is complex and should be in complex or real represent formation!')

    d = x.dim()
    N, Na, Nr = x.size(0), x.size(-3), x.size(-2)

    wshape = [1] * d
    wshape[-3] = Na

    X = ts.fft(x, axis=-3, shift=ftshift)
    phio = th.zeros(wshape, device=x.device, dtype=x.dtype)
    ephi = th.cat((th.cos(phio), th.sin(phio)), dim=-1)
    # print(wshape, phio.min(), phio.max(), ephi.min(), ephi.max())
    So = __fdnorm(ts.ebemulcc(X, ephi), p)
    ii, io, Soi0, Soo0 = 0, 0, 1e13, 1e13
    print(ii, So, Soi0, Soo0)
    while(ii < niter):
        Soo0 = So
        while True:
            Soi0 = So
            print(ii, "===", d)
            for a in range(Na):
                phi1, phi2 = phio.clone().detach(), phio.clone().detach()
                for n in range(N):
                    print(n, N, Na, a, delta)
                    phi1[n, a] += delta
                    phi2[n, a] -= delta
                ephi1 = th.cat((th.cos(phi1), -th.sin(phi1)), dim=-1)
                S1 = __fdnorm(ts.ebemulcc(X, ephi1), p)
                ephi2 = th.cat((th.cos(phi2), -th.sin(phi2)), dim=-1)
                S2 = __fdnorm(ts.ebemulcc(X, ephi2), p)
                print(phi1.shape, phi2.shape, ephi1.shape, ephi2.shape, "]]]")
                print(N, S1, S2, So, S1 - S2, th.sum(ephi1-ephi2), th.sum(phi1-phi2))
                if S1 > So:
                    So = S1
                    phio = phi1.clone().detach()
                    print("111")
                elif S2 > So:
                    print("222")
                    So = S2
                    phio = phi2.clone().detach()
                print(Soi0, So, S1, S2, abs(So - Soi0), io, "+++")
            if abs(So - Soi0) < toli and io > niter:
                break
            io += 1
        print(ii, delta, abs(So - Soo0), tolo, abs(So - Soo0) < tolo, So, Soo0, "So")
        print(ii, delta, abs(So - Soi0), tolo, abs(So - Soi0) < toli, So, Soi0, "So")
        if abs(So - Soo0) > tolo:
            ii += 1
            delta /= 2.
        else:
            break
    ephi = th.cat((th.cos(phio), th.sin(phio)), dim=-1)
    return ts.ifft(ts.ebemulcc(X, ephi)), ephi


def af_ffo(g, w=None, p=2, niter=10, eta=0.5, tol=1e-2, ftshift=False):

    if p < 1:
        raise ValueError('p should be larger than 1!')

    G = ts.fft(g, axis=-3, shift=ftshift)
    Na, Nr = G.size(-3), G.size(-2)
    d = G.dim()
    wshape = [1] * d
    wshape[-3] = Na

    if w is None:
        w = th.ones(wshape, device=G.device, dtype=G.dtype)

    if eta is None:
        eta = ts.PI / 2.

    phi0, i = 1e3, 0
    phi = th.zeros(wshape, device=G.device, dtype=G.dtype)
    diff = th.ones(wshape, device=G.device, dtype=G.dtype)
    # print(i, niter, (phi - phi0).sum(), tol)
    # print(w.shape)
    while(i < niter and (phi - phi0).abs().sum() > tol):
       phi0 = phi
       while(diff.abs().sum() > tol):
          Gabs = th.sqrt(G[..., 0] ** 2 + G[..., 1] ** 2).unsqueeze(-1)
          # print(Gabs.shape, w.shape)
          gamman = th.sum(w * Gabs, axis=-3, keepdim=True)
          print(gamman.min(), gamman.max(), "===")
          # print(gamman.shape, w.shape, Gabs.shape)
          Dphi = (w * p * th.sum((gamman ** (p - 1)) *  Gabs, axis=-2, keepdim=True) / 2.) / \
            (-w * w * p * (p - 1) * th.sum((gamman ** (p - 2)) * (Gabs ** 2), axis=-2, keepdim=True) / 4.)

          # print(Dphi.shape)
          diff = eta * Dphi
          phi = phi - diff
          epa = th.cat((th.cos(phi), th.sin(phi)), axis=-1)
          G = ts.ebemulcc(G, epa)
          print(Dphi.min(), Dphi.max())
          print(diff.min(), diff.max())
          print(phi.min(), phi.max())
       i += 1
       eta /= 2.
       print(i, eta)

    g = ts.ifft(G, axis=-3, shift=ftshift)
    return g
