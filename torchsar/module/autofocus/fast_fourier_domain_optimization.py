#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import torch as th
import torchsar as ts


def _compensation(X, phi, Na, Nr, ftshift=False, isfft=True):
    d = X.dim()
    sizea, sizer = [1] * d, [1] * d
    returns = []
    if Na is not None:
        pa = phi[0]
        sizea[0], sizea[-3], sizea[-1] = pa.size(0), pa.size(1), 2
        epa = th.stack((th.cos(pa), -th.sin(pa)), dim=-1)
        epa = epa.reshape(sizea)

        if isfft:
            X = ts.fft(X, axis=-3, shift=ftshift)
        X = ts.ebemulcc(X, epa)
        returns.append(pa.data)

    if Nr is not None:
        if Na is None:
            pr = phi[0]
        else:
            pr = phi[1]
        sizer[0], sizer[-2], sizer[-1] = pr.size(0), pr.size(1), 2
        epr = th.stack((th.cos(pr), -th.sin(pr)), dim=-1)
        epr = epr.reshape(sizer)

        if isfft:
            X = ts.fft(X, axis=-2, shift=ftshift)
        X = ts.ebemulcc(X, epr)
        returns.append(pr.data)
    return [X] + returns


class AutoFocusFFO(th.nn.Module):

    def __init__(self, Na, Nr, Nb=None, ftshift=False, trainable=True):
        super(AutoFocusFFO, self).__init__()

        if Nb is None:
            Nb = 1

        self.Na = Na
        self.Nr = Nr
        self.Nb = Nb
        self.ftshift = ftshift

        self.barephi = ts.BarePhi(Na, Nr, Nb, pa=None, pr=None, Ma=7, Mr=7, shift=ftshift, trainable=trainable)

    def forward(self, X, isfft=True):

        phi = self.barephi()
        returns = _compensation(X, phi, self.Na, self.Nr, ftshift=self.ftshift, isfft=True)

        return returns

    def imaging(self, Xc):
        if self.Na is not None:
            Xc = ts.ifft(Xc, axis=-3, shift=self.ftshift)
        if self.Nr is not None:
            Xc = ts.ifft(Xc, axis=-2, shift=self.ftshift)
        return Xc

    def get_param(self):
        param = []
        if self.barephi.pa is not None:
            param.append(self.barephi.pa)
        if self.barephi.pr is not None:
            param.append(self.barephi.pr)
        return param

    def param_init(self, pa=None, pr=None):
        self.barephi.param_init(pa, pr)

