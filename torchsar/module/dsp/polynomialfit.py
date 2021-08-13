#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
from torch.nn.parameter import Parameter


class PolyFit(th.nn.Module):

    def __init__(self, c=None, deg=1, trainable=True):
        super(PolyFit, self).__init__()

        if type(deg) is int:
            deg = (0, deg)
        self.deg = deg
        if c is None:
            self.c = Parameter(th.randn(deg[1] - deg[0] + 1, 1), requires_grad=trainable)
        else:
            self.c = Parameter(c, requires_grad=trainable)

    def forward(self, x):
        y = 0.
        for n in range(self.deg[0], self.deg[1] + 1):
            y = y + self.c[n - self.deg[0]] * (x**n)
        return y


if __name__ == '__main__':

    Ns, k, b = 100, 1.2, 3.0
    x = th.linspace(0, 1, Ns)
    t = x * k + b + th.randn(Ns)

    deg = (0, 1)

    polyfit = PolyFit(deg=deg)

    lossfunc = th.nn.MSELoss('mean')
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, polyfit.parameters()), lr=1e-1)

    for n in range(100):
        y = polyfit(x)

        loss = lossfunc(y, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("---Loss %.4f, %.4f, %.4f" % (loss.item(), polyfit.c[0], polyfit.c[1]))
