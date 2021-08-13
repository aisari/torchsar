#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import torch as th
from torchsar.dsp.convolution import fftconv1
from torchsar.module.sharing.matched_filter import RangeMatchedFilter, AzimuthMatchedFilter, AzimuthMatchedFilterLinearFit


class RangeCompress(th.nn.Module):

    def __init__(self, Na, Nr, Tp, Fsr, Kr, Fc, trainable=True, dtype=th.float32):
        super(RangeCompress, self).__init__()
        self.Na = Na
        self.Nr = Nr
        self.Tp = Tp
        self.Fsr = Fsr
        self.Kr = Kr
        self.Fc = Fc
        self.dtype = dtype
        self.rgmf = RangeMatchedFilter(Na, Tp, Fsr, Kr, Fc, trainable=trainable, dtype=dtype)

    def forward(self, X):

        raise TypeError('Not opened yet!')


class AzimuthCompress(th.nn.Module):

    def __init__(self, Na, Nr, Tp, Fsa, Ka, Fc, trainable=True, dtype=th.float32):
        super(AzimuthCompress, self).__init__()
        self.Na = Na
        self.Nr = Nr
        self.Tp = Tp
        self.Fsa = Fsa
        self.Ka = Ka
        self.Fc = Fc
        self.dtype = dtype
        self.azmf = AzimuthMatchedFilter(Nr, Tp, Fsa, Ka, Fc, trainable=trainable, dtype=dtype)

    def forward(self, X):

        raise TypeError('Not opened yet!')


class AzimuthCompressLinearFit(th.nn.Module):

    def __init__(self, Na, Nr, Tp, Fsa, Ka, Fc, trainable=True, dtype=th.float32):
        super(AzimuthCompressLinearFit, self).__init__()
        self.Na = Na
        self.Nr = Nr
        self.Tp = Tp
        self.Fsa = Fsa
        self.Ka = Ka
        self.Fc = Fc
        self.dtype = dtype
        self.azmf = AzimuthMatchedFilterLinearFit(Nr, Tp, Fsa, Ka, Fc, trainable=trainable, dtype=dtype)

    def forward(self, X):

        raise TypeError('Not opened yet!')
