#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchsar as ts


class CICISTA(object):
    def __init__(self, niter=10, mu=0.5, k=None, eps=1e-4):

        self.niter = niter
        self.mu = mu
        self.eps = eps
        self.k = k
        self.t = 0

    def step(self, Y, Xt):

        DX = Y - Xt
        XtmuDX = Xt + self.mu * DX
        AX = th.abs(XtmuDX)
        AX, _ = AX.flatten().sort(descending=True)
        beta = AX[self.k + 1] / self.mu
        X = ts.csoftshrink(XtmuDX, self.mu * beta, None)
        self.r = th.abs(X - Xt).mean()
        self.t += 1
        return X, self.r

    def optimize(self, Y):

        X = 0
        for t in range(self.niter):
            X, r = self.step(Y, X)
            if r < self.eps:
                break
        return X, self.r
