#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-16 10:28:33
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th
from torchsar.utils.const import *
from torchsar.dsp.normalsignals import rect


def sar_tran(t, Tp, Kr, Fc, A=1.):

    return A * rect(t / Tp) * th.exp(2j * PI * Fc * t + 1j * PI * Kr * t**2)


def sar_recv(t, tau, Tp, Kr, Fc, A=1.):

    t = t - tau
    return A * rect(t / Tp) * th.exp(2j * PI * Fc * t + 1j * PI * Kr * t**2)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from torch.fft import fft, fftshift

    Kr = 40e+12
    Tp = 2.5e-06
    Br = abs(Kr) * Tp

    alpha = 1.24588  # 1.1-1.4
    Fsr = alpha * Br
    # Fc = 5.3e9
    Fc = 0.

    Tsr = 1.2 * Tp
    Nsr = int(Fsr * Tsr)
    t = th.linspace(-Tsr / 2., Tsr / 2, Nsr)
    f = th.linspace(-Fsr / 2., Fsr / 2, Nsr)

    St = sar_tran(t, Tp, Kr, Fc)

    Yt = fftshift(fft(fftshift(St, dim=0), dim=0), dim=0)

    plt.figure(1)
    plt.subplot(221)
    plt.plot(t * 1e6, th.real(St))
    plt.plot(t * 1e6, th.abs(St))
    plt.grid()
    plt.legend({'Real part', 'Amplitude'})
    plt.title('Matched filter')
    plt.xlabel('Time/μs')
    plt.ylabel('Amplitude')
    plt.subplot(222)
    plt.plot(t * 1e6, th.angle(St))
    plt.grid()
    plt.subplot(223)
    plt.plot(f, th.abs(Yt))
    plt.grid()
    plt.subplot(224)
    plt.plot(f, th.angle(Yt))
    plt.grid()
    plt.show()
