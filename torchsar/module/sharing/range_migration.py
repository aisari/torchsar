#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import torch as th
import torchsar as ts
from torch.nn.parameter import Parameter


class RangeMigrationCorrection(th.nn.Module):

    def __init__(self, Na, Nr, R0, Vr, Fc, Fsa, Fsr, D=None):
        super(RangeMigrationCorrection, self).__init__()
        self.Na = Na
        self.Nr = Nr
        self.R0 = R0
        self.Vr = Vr
        self.Fc = Fc
        self.Fsa = Fsa
        self.Fsr = Fsr

        tnear = 2. * R0 / ts.C
        tfar = tnear + Nr / Fsr
        tr = th.ones(self.Na, 1) * th.linspace(tnear, tfar, Nr)
        self.tr = Parameter(tr, requires_grad=False)

        if D is None:
            Wl = ts.C / Fc
            fa = ts.fftfreq(Na, Fsa, norm=False, shift=False).reshape(Na, 1)
            D = th.sqrt(1.0 - (Wl * fa / (2.0 * Vr)) ** 2)
        self.D = Parameter(D, requires_grad=False)
        self.interp = ts.Interp1()

    def forward(self, X):
        raise TypeError('Not opened yet!')


if __name__ == '__main__':

    Na, Nr = 8, 8
    X = th.randn(Na, Nr, 2)
    print(X)

    rcmclayer = RangeMigrationCorrection(Na, Nr)

    X = rcmclayer(X)
    print(X)
