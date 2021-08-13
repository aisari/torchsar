#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 22:29:14
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

import torch as th
import numpy as np
import matplotlib.pyplot as plt


def center_dominant_scatters(x, axis=-2, isplot=False):
    """Center the dominant scatters.

    Parameters
    ----------
    x : tensor
        Complex SAR image tensor.
    axis : int, optional
        The axis to be centered in.
    isplot : bool, optional
        Whether to plot.

    Returns
    -------
    tensor
        Centered.

    Raises
    ------
    TypeError
        Intput must be complex image!
    """
    if th.is_complex(x):
        cplxflag = True
        pass
    elif x.size(-1) == 2:
        cplxflag = False
        x = x[..., 0] + 1j * x[..., 1]
    else:
        raise TypeError('Intput must be complex image!')

    if x.dim() == 2:
        x = x.unsqueeze(0)

    Nx = x.size(axis)
    N, Na, Nr = x.size()
    maxpos = th.argmax(x.abs(), axis=axis)  # N-Na-Nr

    # ---Step 1: Samples selection
    midpos = int(Nx / 2.)
    Nshift = midpos - maxpos

    # ---Step 2: Circular shifting in azimuth
    z = th.zeros_like(x)
    if axis == -2 or axis == 1:
        for n in range(N):
            for r in range(Nr):
                z[n, :, r] = th.roll(x[n, :, r], Nshift[n, r].item())
    if axis == -1 or axis == 2:
        for n in range(N):
            for a in range(Na):
                z[n, a, :] = th.roll(x[n, a, :], Nshift[n, a].item())

    if isplot:
        plt.figure()
        plt.imshow(th.abs(z[0]).numpy(), cmap='gray')
        plt.xlabel('Range/samples')
        plt.ylabel('Azimuth/samples')
        plt.title('Centered')
        plt.show()

    if cplxflag is False:
        z = th.view_as_real(z)

    return z


def window_data(z, win=None, axis=-2, isplot=False):
    showidx = -1

    if th.is_complex(z):
        cplxflag = True
        pass
    elif z.size(-1) == 2:
        cplxflag = False
        z = z[..., 0] + 1j * z[..., 1]
    else:
        raise TypeError('Intput must be complex image!')

    if z.dim() == 2:
        z = z.unsqueeze(0)

    N, Na, Nr = z.size()
    Nx = z.size(axis)
    midpos = int(Nx / 2.)
    if axis == -2 or axis == 1:
        axisw, axiso = -2, -1
        winshape = [Nx, 1]
    if axis == -1 or axis == 2:
        axisw, axiso = -1, -2
        winshape = [1, Nx]

    # ---Step 3: Windowing
    ncoh_avg_win = th.sum(z.conj() * z, axis=axiso)  # N-Nx
    ncoh_avg_win_db20 = 20. * th.log10(th.abs(ncoh_avg_win))  # N-Nx
    win_cutoff = th.mean(ncoh_avg_win_db20, axis=-1)  # N-1

    for n in range(N):
        leftidx = midpos
        rightidx = midpos
        for i in range(midpos, 0, -1):
            if ncoh_avg_win_db20[n, i] < win_cutoff[n]:
                leftidx = i
                break
        for i in range(midpos, Nx, 1):
            if ncoh_avg_win_db20[n, i] < win_cutoff[n]:
                rightidx = i
                break
        ncoh_avg_win = th.zeros(Nx, device=z.device)
        ncoh_avg_win[leftidx:rightidx] = 1.
        z[n] = z[n] * (ncoh_avg_win.reshape(winshape))

    if isplot:
        plt.figure()
        plt.grid()
        plt.plot(ncoh_avg_win_db20[showidx].numpy(), 'b')
        plt.plot(win_cutoff[showidx] * np.ones(Na), 'r')
        plt.plot(ncoh_avg_win.numpy(), 'g')
        plt.legend(['non-coherent', 'cutoff', 'window'])
        plt.xlabel('Sample index in azimuth direction')
        plt.ylabel('Amplitude/dB')
        plt.title('Window')
        plt.show()

    if isplot:
        plt.imshow(th.abs(z[showidx]).numpy(), cmap='gray')
        plt.xlabel('Range/samples')
        plt.ylabel('Azimuth/samples')
        plt.title('Windowed')
        plt.show()

    if cplxflag is False:
        z = th.view_as_real(z)

    return z
