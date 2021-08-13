#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import torch as th
from torch.nn.parameter import Parameter
from torchsar.dsp.ffts import fft, ifft
from torchsar.base.mathops import ebemulcc


class AutoFocus(th.nn.Module):

    def __init__(self, Na, Nr, pa=None, pr=None, ftshift=False, trainable=True):
        super(AutoFocus, self).__init__()

        self.Na = Na
        self.Nr = Nr
        self.ftshift = ftshift

        if Na is not None:
            if pa is None:
                pa = th.zeros(1, Na)
            self.pa = Parameter(pa, requires_grad=trainable)

        if Nr is not None:
            if pr is None:
                pr = th.zeros(1, Nr)
            self.pr = Parameter(pr, requires_grad=trainable)

    def forward(self, X, isfft=True):
        d = X.dim()
        sizea, sizer = [1] * d, [1] * d
        if self.Na is not None:
            sizea[-3], sizea[-1] = self.pa.size(1), 2
            epa = th.stack((th.cos(self.pa), -th.sin(self.pa)), dim=-1)
            epa = epa.reshape(sizea)

            if isfft:
                X = fft(X, axis=-3, shift=self.ftshift)
            X = ebemulcc(X, epa)
            X = ifft(X, axis=-3, shift=self.ftshift)

        if self.Nr is not None:
            sizer[-3], sizer[-1] = self.pr.size(1), 2
            epr = th.stack((th.cos(self.pr), -th.sin(self.pr)), dim=-1)
            epr = epr.reshape(sizer)

            if isfft:
                X = fft(X, axis=-2, shift=self.ftshift)
            X = ebemulcc(X, epr)
            X = ifft(X, axis=-2, shift=self.ftshift)

        return X
