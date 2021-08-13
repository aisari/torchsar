#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import torch as th
import torchsar as ts


def focus(x, pa=None, pr=None, isfft=True, ftshift=True):
    r"""Focus image with given phase error

    Focus image in azimuth by

    .. math::
        Y(k, n_r)=\sum_{n_a=0}^{N_a-1} X(n_a, n_r) \exp \left(-j \varphi_{n_a}\right) \exp \left(-j \frac{2 \pi}{N_a} k n_a\right)

    where, :math:`\varphi_{n_a}` is the estimated azimuth phase error of the :math:`n_a`-th azimuth line, :math:`y(k, n_r)` is
    the focused pixel.

    The focus method in range is the same as azimuth.

    Args:
        x (Tensor): Defocused image data :math:`{\mathbf X} \in{\mathbb C}^{N\times N_c\times N_a\times N_r}` or
            :math:`{\mathbf X} \in{\mathbb R}^{N\times N_a\times N_r\times 2}` or
            :math:`{\mathbf X} \in{\mathbb R}^{N\times N_c\times N_a\times N_r\times 2}`.
        pa (Tensor, optional): Focus parameters in azimuth, phase error in rad unit. (the default is None, not focus)
        pr (Tensor, optional): Focus parameters in range, phase error in rad unit. (the default is None, not focus)
        isfft (bool, optional): Is need do fft (the default is True)
        ftshift (bool, optional): Is shift zero frequency to center when do fft/ifft/fftfreq (the default is True)

    Returns:
        (Tensor): A tensor of focused images.

    Raises:
        TypeError: :attr:`x` is complex and should be in complex or real represent formation!
    """

    if type(x) is not th.Tensor:
        x = th.tensor(x)

    if th.is_complex(x):
        # N, Na, Nr = x.size(0), x.size(-2), x.size(-1)
        x = th.view_as_real(x)
        crepflag = True
    elif x.size(-1) == 2:
        # N, Na, Nr = x.size(0), x.size(-3), x.size(-2)
        crepflag = False
    else:
        raise TypeError('x is complex and should be in complex or real represent formation!')

    d = x.dim()
    sizea, sizer = [1] * d, [1] * d

    if pa is not None:
        sizea[0], sizea[-3], sizea[-1] = pa.size(0), pa.size(1), 2
        epa = th.stack((th.cos(pa), -th.sin(pa)), dim=-1)
        epa = epa.reshape(sizea)

        if isfft:
            x = ts.fft(x, axis=-3, shift=ftshift)
        x = ts.ebemulcc(x, epa)
        x = ts.ifft(x, axis=-3, shift=ftshift)

    if pr is not None:
        sizer[0], sizer[-2], sizer[-1] = pr.size(0), pr.size(1), 2
        epr = th.stack((th.cos(pr), -th.sin(pr)), dim=-1)
        epr = epr.reshape(sizer)

        if isfft:
            x = ts.fft(x, axis=-2, shift=ftshift)
        x = ts.ebemulcc(x, epr)
        x = ts.ifft(x, axis=-2, shift=ftshift)

    if crepflag:
        x = th.view_as_complex(x)

    return x


def defocus(x, pa=None, pr=None, isfft=True, ftshift=True):
    r"""Defocus image with given phase error

    Defocus image in azimuth by

    .. math::
        Y(k, n_r)=\sum_{n_a=0}^{N_a-1} X(n_a, n_r) \exp \left(j \varphi_{n_a}\right) \exp \left(-j \frac{2 \pi}{N_a} k n_a\right)

    where, :math:`\varphi_{n_a}` is the estimated azimuth phase error of the :math:`n_a`-th azimuth line, :math:`y(k, n_r)` is
    the focused pixel.

    The defocus method in range is the same as azimuth.

    Args:
        x (Tensor): Focused image data :math:`{\mathbf X} \in{\mathbb C}^{N\times N_c\times N_a\times N_r}` or
            :math:`{\mathbf X} \in{\mathbb R}^{N\times N_a\times N_r\times 2}` or
            :math:`{\mathbf X} \in{\mathbb R}^{N\times N_c\times N_a\times N_r\times 2}`.
        pa (Tensor, optional): Defocus parameters in azimuth, phase error in rad unit. (the default is None, not focus)
        pr (Tensor, optional): Defocus parameters in range, phase error in rad unit. (the default is None, not focus)
        isfft (bool, optional): Is need do fft (the default is True)
        ftshift (bool, optional): Is shift zero frequency to center when do fft/ifft/fftfreq (the default is True)

    Returns:
        (Tensor): A tensor of defocused images.

    Raises:
        TypeError: :attr:`x` is complex and should be in complex or real represent formation!
    """

    if type(x) is not th.Tensor:
        x = th.tensor(x)

    if th.is_complex(x):
        # N, Na, Nr = x.size(0), x.size(-2), x.size(-1)
        x = th.view_as_real(x)
        crepflag = True
    elif x.size(-1) == 2:
        # N, Na, Nr = x.size(0), x.size(-3), x.size(-2)
        crepflag = False
    else:
        raise TypeError('x is complex and should be in complex or real represent formation!')

    d = x.dim()
    sizea, sizer = [1] * d, [1] * d

    if pa is not None:
        sizea[0], sizea[-3], sizea[-1] = pa.size(0), pa.size(1), 2
        epa = th.stack((th.cos(pa), th.sin(pa)), dim=-1)
        epa = epa.reshape(sizea)

        if isfft:
            x = ts.fft(x, axis=-3, shift=ftshift)
        x = ts.ebemulcc(x, epa)
        x = ts.ifft(x, axis=-3, shift=ftshift)

    if pr is not None:
        sizer[0], sizer[-2], sizer[-1] = pr.size(0), pr.size(1), 2
        epr = th.stack((th.cos(pr), th.sin(pr)), dim=-1)
        epr = epr.reshape(sizer)

        if isfft:
            x = ts.fft(x, axis=-2, shift=ftshift)
        x = ts.ebemulcc(x, epr)
        x = ts.ifft(x, axis=-2, shift=ftshift)

    if crepflag:
        x = th.view_as_complex(x)

    return x


if __name__ == '__main__':

    n = 8
