#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torch.nn.functional as F
import torchsar as ts


class FFTLayer1d(th.nn.Module):

    def __init__(self, nfft=None):
        super(FFTLayer1d, self).__init__()
        self.nfft = nfft

    def forward(self, x):
        n, d, _ = x.size()
        if self.nfft is None:
            self.nfft = d
        if d != self.nfft:
            x = F.pad(x, [0, self.nfft - d, 0], mode='constant', value=0)
        # y = th.fft.fft(x, n=None, dim=0, norm=None)
        y = ts.fft(x, n, axis=0, norm=None)

        return y


if __name__ == '__main__':

    import numpy as np
    import torch as th
    import matplotlib.pyplot as plt

    PI = np.pi
    f0 = 100
    Fs = 1000
    Ts = 0.1
    Ns = int(Fs * Ts)

    f = np.linspace(0., Fs, Ns)
    t = np.linspace(0, Ts, Ns)
    x_np = np.cos(2. * PI * f0 * t) + 1j * np.sin(2. * PI * f0 * t)

    device = th.device('cuda:0')
    # x_th = th.tensor(x_np, dtype=th.complex64)
    x_th = th.tensor([x_np.real, x_np.imag], dtype=th.float32).transpose(1, 0)
    x_th = x_th.to(device)
    print(x_th.shape, type(x_th))

    x_ths = th.tensor([x_th.cpu().numpy(), x_th.cpu().numpy(),
                       x_th.cpu().numpy()], dtype=th.float32)

    print(x_ths.shape)

    fftlayer = FFTLayer1d()
    ys = fftlayer.forward(x_ths)
    ys = th.abs(ys[:, :, 0] + 1j * ys[:, :, 1]).cpu()

    plt.figure()
    plt.subplot(131)
    plt.plot(f, ys[0])
    plt.grid()
    plt.subplot(132)
    plt.plot(f, ys[1])
    plt.grid()
    plt.subplot(133)
    plt.plot(f, ys[2])
    plt.grid()
    plt.show()
