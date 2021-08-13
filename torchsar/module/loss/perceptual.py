#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchsar as ts


class RandomProjectionLoss(th.nn.Module):
    r"""RandomProjection loss



    """

    def __init__(self, mode='real', baseloss='MSE', channels=[3, 32], kernel_sizes=[(3, 3)], activations=['ReLU'], reduction='mean', device='cpu'):
        super(RandomProjectionLoss, self).__init__()
        self.mode = mode
        self.reduction = reduction

        nLayers = len(kernel_sizes)
        self.nLayers = nLayers

        net = []
        if mode == 'real':
            conv = th.nn.Conv2d
            bn = th.nn.BatchNorm2d
        if mode == 'complex':
            conv = ts.ComplexConv2d
            bn = ts.ComplexBatchNorm2d

        for n in range(nLayers):
            net.append(conv(channels[n], channels[n + 1], kernel_sizes[n]))
            net.append(bn(channels[n + 1]))
            net.append(eval('th.nn.' + activations[n])())
            self.N = 3

        self.rpf = th.nn.ModuleList(net)
        if baseloss in ['MSE', 'mse']:
            self.baseloss = th.nn.MSELoss(reduction=reduction)
        if baseloss in ['MAE', 'mae']:
            self.baseloss = th.nn.MAELoss(reduction=reduction)

        self.weight_init()
        self.rpf = self.rpf.to(device)

    def forward(self, P, G):

        with th.no_grad():
            loss = 0.
            for n in range(self.nLayers * self.N):
                P, G = self.rpf[n](P), self.rpf[n](G)
                if n % self.N == self.N - 1:
                    loss += self.baseloss(P, G)

            if self.reduction == 'mean':
                loss /= self.nLayers
            if self.reduction == 'sum':
                pass
        return loss

    def weight_init(self):

        for n in range(self.nLayers):
            if self.mode in ['Complex', 'complex']:
                th.nn.init.orthogonal_(self.rpf[n * self.N].convr.weight, th.nn.init.calculate_gain('leaky_relu'))
                th.nn.init.orthogonal_(self.rpf[n * self.N].convi.weight, th.nn.init.calculate_gain('leaky_relu'))
            else:
                th.nn.init.orthogonal_(self.rpf[n * self.N].weight, th.nn.init.calculate_gain('leaky_relu'))

        for param in self.parameters():
            param.requires_grad = False


if __name__ == '__main__':

    loss_func = RandomProjectionLoss(mode='real', baseloss='MSE', channels=[3, 4], kernel_sizes=[(3, 3)], activations=['LeakyReLU'], reduction='mean')
    P = th.randn(4, 3, 64, 64)
    G = th.randn(4, 3, 64, 64)
    S = loss_func(P, G)
    print(S)

    loss_func = RandomProjectionLoss(mode='complex', baseloss='MSE', channels=[3, 4], kernel_sizes=[(3, 3)], activations=['LeakyReLU'], reduction='mean')
    P = th.randn(4, 3, 64, 64, 2)
    G = th.randn(4, 3, 64, 64, 2)
    S = loss_func(P, G)
    print(S)
