#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


class DataConsistency2d(th.nn.Module):

    def __init__(self, ftaxis=(-2, -1), mixrate=1.0, isfft=True):
        super(DataConsistency2d, self).__init__()
        self.ftaxis = ftaxis
        self.mixrate = mixrate
        self.isfft = isfft

    def forward(self, x, y, mask):
        d = x.dim()
        maskshape = [1] * d
        for a in self.ftaxis:
            maskshape[a] = x.shape[a]
        mask = mask * self.mixrate
        mask = mask.reshape(maskshape)
        if self.isfft:
            xf = th.fft.fft2(x, s=None, dim=self.ftaxis, norm=None)
            yf = th.fft.fft2(y, s=None, dim=self.ftaxis, norm=None)

        xf = xf * mask
        yf = yf * (1.0 - mask)

        return th.fft.ifft2(xf + yf, s=None, dim=self.ftaxis, norm=None)


if __name__ == '__main__':

    N, C, H, W = 5, 2, 3, 4
    x = th.randn(N, C, H, W)
    y = th.randn(N, C, H, W)
    mask = th.rand(H, W)
    mask[mask < 0.5] = 0
    mask[mask > 0.5] = 1
    # mask = th.ones(H, W)

    dc = DataConsistency2d(ftaxis=(-2, -1), mixrate=1., isfft=True)

    z = dc(x, y, mask)

    print(x, 'x')
    print(y, 'y')
    print(z.abs(), 'z')
