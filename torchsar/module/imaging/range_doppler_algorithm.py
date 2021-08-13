#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import time
import torch as th
import torchsar as ts


class LRDAnet(th.nn.Module):

    def __init__(self, Na, Nr, Fsa, Fsr, Ta, Tp, Ka, Kr, fdc, Fc, R0, Vr, La):
        super(LRDAnet, self).__init__()
        self.Na = Na
        self.Nr = Nr
        self.Fsa = Fsa
        self.Fsr = Fsr
        self.Ta = Ta
        self.Tp = Tp
        self.Ka = Ka
        self.Kr = Kr
        self.fdc = fdc
        self.Fc = Fc
        self.R0 = R0
        self.Vr = Vr
        self.La = La

        Noff, Wl = th.linspace(0, Nr, Nr), ts.C / Fc
        Rp = ts.min_slant_range(R0, Fsr, Noff)

        if Ka is None:
            Ka = ts.dre_geo(Wl, Vr, Rp)

        if Ta is None:
            FPa = ts.azimuth_footprint(Rp, Wl, La)
            Ta = th.mean(FPa).item() / Vr

        self.rclayer = ts.RangeCompress(Na, Nr, Tp, Fsr, Kr, Fc=0., trainable=[False, False], dtype=th.float64)
        self.rcmclayer = ts.RangeMigrationCorrection(Na, Nr, R0=R0, Vr=Vr, Fc=Fc, Fsa=Fsa, Fsr=Fsr, D=None)
        self.aclayer = ts.AzimuthCompressLinearFit(Na, Nr, Ta, Fsa, Ka, fdc, trainable=[True, False], dtype=th.float32)
        self.aflayer = ts.AutoFocus(Na, Nr, pa=None, pr=None, trainable=False)

    def forward(self, X):
        # X --> Na-Nr-2

        X = self.rclayer(X)
        X = self.rcmclayer(X)
        X = self.aclayer(X)
        X = self.aflayer(X)

        return X

    def focus(self, X, pa=None, pr=None):
        # X --> N-1-Na-Nr-2
        # pa --> N-Na
        # pr --> N-Nr

        if pa is None and pr is None:
            return X

        if pa is not None:
            X = ts.fft(X, nfft=None, axis=2, norm=False)
            pa = pa.reshape(pa.size(0), 1, int(pa.numel() / pa.size(0)), 1)
            epa = th.stack((th.cos(pa), th.sin(pa)), dim=-1)
            X = ts.ebemulcc(X, epa)
        if pr is not None:
            X = ts.fft(X, nfft=None, axis=3, norm=False)
            pr = pr.reshape(pr.size(0), 1, 1, int(pr.numel() / pr.size(0)))
            epr = th.stack((th.cos(pr), th.sin(pr)), dim=-1)
            X = ts.ebemulcc(X, epr)
        if pa is not None:
            X = ts.ifft(X, nfft=None, axis=2, norm=False)
        if pr is not None:
            X = ts.ifft(X, nfft=None, axis=3, norm=False)

        return X


if __name__ == "__main__":

    epochs = 10
    learning_rate = 1e-3
    device = th.device('cuda:1' if th.cuda.is_available() else 'cpu')
    # device = th.device('cpu')
    print(device)

    Na, Nr = 512, 512
    net = LRDAnet(Na, Nr)
    net.to(device=device)

    X = th.randn((4, Na, Nr, 2), requires_grad=False)
    X = X.to(device)

    loss_func = tht.ShannonEntropy(reduction='mean')
    optimizer = th.optim.Adam(net.parameters(), lr=learning_rate)

    for k in range(epochs):
        tstart = time.time()

        pa, Y = net.forward(X)
        loss = loss_func(Y)

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        tend = time.time()

        print("---loss: %s, time: %ss" % (loss.item(), tend - tstart))
