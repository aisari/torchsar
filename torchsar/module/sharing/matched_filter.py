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


class RangeMatchedFilter(th.nn.Module):

    def __init__(self, Na, Tp, Fsr, Kr, Fc, trainable=True, dtype=th.float64):
        super(RangeMatchedFilter, self).__init__()
        self.Na = Na
        self.Tp = Tp
        self.dtype = dtype  # filters with float64 is better then float32

        if type(trainable) is bool:
            trainable = [trainable] * 2
        self.Fsr = Fsr
        self.Kr = Parameter(th.tensor(Kr, dtype=dtype), requires_grad=trainable[0])
        self.Fc = Parameter(th.tensor(Fc, dtype=dtype), requires_grad=trainable[1])
        self.N = round(Fsr * Tp)

        t = th.linspace(-Tp / 2., Tp / 2., self.N).reshape(1, self.N)
        self.t = Parameter(t, requires_grad=False)

    def forward(self):

        # mf = th.exp(2j * ts.PI * self.Fc * self.t - 1j * ts.PI * self.Kr * self.t**2)
        # mf = th.exp(2j * ts.PI * self.Fc * self.t - 1j * ts.PI * ((self.t**2) * self.Kr))

        p = ts.PI * (2. * self.Fc * self.t - ((self.t**2) * self.Kr))
        mf = th.stack((th.cos(p), th.sin(p)), dim=-1)  # 1-N-2
        # mf = th.stack((th.cos(p) * ts.rect(self.t / self.Tp), th.sin(p) *
        # ts.rect(self.t / self.Tp)), dim=-1)  # 1-N-2
        return mf


class AzimuthMatchedFilter(th.nn.Module):

    def __init__(self, Nr, Tp, Fsa, Ka, Fc, trainable=True, dtype=th.float32):
        super(AzimuthMatchedFilter, self).__init__()
        self.Nr = Nr
        self.Tp = Tp
        self.dtype = dtype  # filters with float64 is better then float32

        if type(trainable) is bool:
            trainable = [trainable] * 2

        if type(Ka) is float:
            Ka = th.ones(1, Nr) * Ka
        else:
            Ka = th.tensor(Ka, dtype=dtype).reshape(1, Nr)

        self.Fsa = Fsa
        self.Ka = Parameter(Ka, requires_grad=trainable[0])
        self.Fc = Parameter(th.tensor(Fc, dtype=dtype), requires_grad=trainable[1])
        self.N = round(Fsa * Tp)

        t = th.linspace(-Tp / 2., Tp / 2., self.N).reshape(self.N, 1)
        self.t = Parameter(t, requires_grad=False)

    def forward(self):

        raise TypeError('Not opened yet!')


class AzimuthMatchedFilterLinearFit(th.nn.Module):

    def __init__(self, Nr, Tp, Fsa, Ka, Fc, trainable=True, dtype=th.float32):
        super(AzimuthMatchedFilterLinearFit, self).__init__()
        self.Nr = Nr
        self.Tp = Tp
        self.dtype = dtype  # filters with float64 is better then float32

        if type(trainable) is bool:
            trainable = [trainable] * 2

        if type(Ka) is float:
            Ka = th.ones(1, Nr) * Ka
        else:
            Ka = Ka.reshape(1, Nr)

        x = th.linspace(0, Nr, Nr).reshape(1, Nr) / Nr
        wa = ts.polyfit(x, Ka, deg=1).reshape(1, 2)
        xa = th.cat((th.ones(1, Nr), x), axis=0)

        self.Fsa = Fsa
        self.wa = Parameter(wa, requires_grad=True)
        self.xa = Parameter(xa, requires_grad=False)
        self.Ka = Parameter(Ka, requires_grad=False)
        self.Fc = Parameter(th.tensor(Fc, dtype=dtype), requires_grad=trainable[1])
        self.N = round(Fsa * Tp)

        t = th.linspace(-Tp / 2., Tp / 2., self.N).reshape(self.N, 1)
        self.t = Parameter(t, requires_grad=False)

    def forward(self):

        raise TypeError('Not opened yet!')


if __name__ == '__main__':

    pass
