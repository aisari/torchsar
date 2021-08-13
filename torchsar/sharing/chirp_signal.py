#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-06 10:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import torch as th

from torchsar.utils.const import *
from torchsar.dsp.normalsignals import rect


def chirp_tran(t, Tp, K, Fc, a=1.):

    return a * rect(t / Tp) * th.exp(2j * PI * Fc * t + 1j * PI * K * t**2)


def chirp_recv(t, Tp, K, Fc, a=1., g=1., r=1e3):
    Sr = 0.
    for gi, ri in zip(g, r):
        tau = 2. * ri / C
        tt = t - tau  # do not use t -= tau, this will change t out this function
        Sr += gi * rect(tt / Tp) * th.exp(2j * PI * Fc * tt + 1j * PI * K * tt**2)
    return Sr


class Chirp(th.nn.Module):

    def __init__(self, Tp, K, Fc=0., a=1.):
        self.Tp = Tp
        self.K = K
        self.Fc = Fc
        self.a = a
        self.C = 299792458.
        self.PI = 3.141592653589793

    def tran(self, t):
        return self.a * rect(t / self.Tp) * th.exp(2j * self.PI * self.Fc * t + 1j * self.PI * self.K * t**2)

    def recv(self, t, g, r):

        Sr = 0.
        for gi, ri in zip(g, r):
            td = 2. * ri / self.C
            tt = t - td  # do not use t -= td, this will change t out this function
            Sr += self.a * gi * rect(tt / self.Tp) * th.exp(2j * self.PI * self.Fc * tt + 1j * self.PI * self.K * tt**2)

        return Sr


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from torch.fft import fft, fftshift

    # ===Generate tansmitted and recieved signals
    # ---Setting parameters
    R = [1.e3, 2.e3, 3.e3]
    G = [0.5, 1.0, 0.8]

    EPS = 2.2e-32
    K = 4.1e+11
    Tp = 37.0e-06
    Br = abs(K) * Tp

    alp = 1.24588  # 1.1-1.4
    Fsr = alp * Br
    Fc = 5.3e9
    Fc = 0.

    Tsr = 2.1 * Tp
    Nsr = int(Fsr * Tsr)
    t = th.linspace(-Tsr / 2., Tsr / 2, Nsr)
    f = th.linspace(-Fsr / 2., Fsr / 2, Nsr)

    # ---Transmitted signal
    St = chirp_tran(t, Tp, K, Fc, a=1.)

    # ---Recieved signal
    Sr = chirp_recv(t, Tp, K, Fc, a=1., g=G, r=R)

    chirp = Chirp(Tp=Tp, K=K, Fc=Fc, a=1.)

    St = chirp.tran(t)
    Sr = chirp.recv(t, g=G, r=R)

    plt.figure()
    plt.subplot(221)
    plt.plot(t * 1e6, th.real(St))
    plt.plot(t * 1e6, th.imag(St))
    plt.xlabel('Time/us')
    plt.legend(['real', 'imag'])
    plt.subplot(222)
    plt.plot(t * 1e6, th.angle(St))
    plt.xlabel('Time/us')
    plt.subplot(223)
    plt.plot(t * 1e6, th.real(Sr))
    plt.plot(t * 1e6, th.imag(Sr))
    plt.xlabel('Time/us')
    plt.legend(['real', 'imag'])
    plt.subplot(224)
    plt.plot(t * 1e6, th.angle(Sr))
    plt.xlabel('Time/us')
    plt.show()


    # ---Frequency domain
    Yt = fftshift(fft(fftshift(St, dim=0), dim=0), dim=0)
    Yr = fftshift(fft(fftshift(Sr, dim=0), dim=0), dim=0)

    # ---Plot signals
    plt.figure(figsize=(10, 8))
    plt.subplot(221)
    plt.plot(t * 1e6, th.real(St))
    plt.grid()
    plt.title('Real part')
    plt.xlabel('Time/μs')
    plt.ylabel('Amplitude')
    plt.subplot(222)
    plt.plot(t * 1e6, th.imag(St))
    plt.grid()
    plt.title('Imaginary part')
    plt.xlabel('Time/μs')
    plt.ylabel('Amplitude')
    plt.subplot(223)
    plt.plot(f, th.abs(Yt))
    plt.grid()
    plt.title('Spectrum')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Amplitude')
    plt.subplot(224)
    plt.plot(f, th.angle(Yt))
    plt.grid()
    plt.title('Spectrum')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Phase')
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98, top=0.96, wspace=0.19, hspace=0.25)

    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplot(221)
    plt.plot(t * 1e6, th.real(Sr))
    plt.grid()
    plt.title('Real part')
    plt.xlabel('Time/μs')
    plt.ylabel('Amplitude')
    plt.subplot(222)
    plt.plot(t * 1e6, th.imag(Sr))
    plt.grid()
    plt.title('Imaginary part')
    plt.xlabel('Time/μs')
    plt.ylabel('Amplitude')
    plt.subplot(223)
    plt.plot(f, th.abs(Yr))
    plt.grid()
    plt.title('Spectrum')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Amplitude')
    plt.subplot(224)
    plt.plot(f, th.angle(Yr))
    plt.grid()
    plt.title('Spectrum')
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Phase')
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98, top=0.96, wspace=0.19, hspace=0.25)

    plt.show()
