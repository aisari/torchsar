#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th
from skimage import filters
from torchsar.utils.const import EPS
import matplotlib.pyplot as plt


def extract_targets(x, region=None, thresh=None, caxis=None, isshow=False):
    """Extracts the pixels of targetss

    Parameters
    ----------
    x : tensor or numpy array
        The input image, if it's complex-valued, it's amplitude is used.
    region : list or None, optional
        The sub-region that contains targets.
    thresh : float or None, optional
        The threshold targets.
    caxis : int or None, optional
        Specifies the complex axis of :attr:`x`. ``None`` --> complex or real
    isshow : bool, optional
        Show pixels of the extracted targets?
    """

    if th.is_complex(x):
        caxis = None
    if caxis is not None:
        x = x.pow(2).sum(caxis).sqrt()
    else:
        x = x.abs()
    x = x.numpy()

    x = x[region[0]:region[2], region[1]:region[3]]

    thresh = filters.threshold_otsu(x) if thresh is None else thresh

    idx = np.where(x > thresh)

    if isshow:
        plt.figure
        plt.subplot(121)
        plt.imshow(x)
        y = np.zeros(x.shape)
        y[idx] = 1
        plt.subplot(122)
        plt.imshow(x * y)
        plt.show()
    idx = list(idx)
    idx[0] += region[0]
    idx[1] += region[1]

    return tuple(idx), thresh


def tbr(x, targets, region=None, caxis=None, isshow=False):
    r"""Target-to-Background Ratio (TBR)

    .. math::
        \begin{align}
        {\rm TBR} = 20{\rm log}_{10}\left(\frac{{\rm max}_{i\in{\mathbb T}}(|{\bf X}_i|)}{{(1/N_{\mathbb B})}\Sigma_{j\in \mathbb B}|{\bf X}_j|)}\right)
        \label{equ:TBR}
        \end{align}

    Parameters
    ----------
    x : tensor
        The input image, if it's complex-valued, it's amplitude is used.
    targets : list or tuple
        The targets pixels. ([r1, r2, ...], [c1, c2, ...]])
    region : list, tuple or None, optional
        The region for computing TBR. ([lefttop, rightbottom])
    caxis : int or None, optional
        Specifies the complex axis of :attr:`x`. ``None`` --> complex or real
    isshow : bool, optional
        Show target mask? (default: False)
    """

    if th.is_complex(x):
        caxis = None
    if caxis is not None:
        x = x.pow(2).sum(caxis).sqrt()
    else:
        x = x.abs()

    TGM = th.zeros(x.shape, dtype=th.int8)
    TGM[targets] = 1
    TGM = TGM[region[0]:region[2], region[1]:region[3]].to(x.device)
    x = x[region[0]:region[2], region[1]:region[3]]

    # mask of BG
    BGM = 1 - TGM

    # pixel number of bgs
    NB = BGM.sum()

    R = th.max(x * TGM) / (((1 / NB) * th.sum(x * BGM)) + EPS)

    TBR = 20 * th.log10(R).item()

    if isshow:
        plt.figure
        plt.subplot(131)
        plt.imshow(x)
        plt.subplot(132)
        plt.imshow(TGM)
        plt.subplot(133)
        plt.imshow(x * TGM)
        plt.show()

    return TBR


def tbr2(X, tgrs, subrs=None, isshow=False):
    r"""Target-to-Background Ratio (TBR)

    .. math::
        \begin{align}
        {\rm TBR} = 20{\rm log}_{10}\left(\frac{{\rm max}_{i\in{\mathbb T}}(|{\bf X}_i|)}{{(1/N_{\mathbb B})}\Sigma_{j\in \mathbb B}|{\bf X}_j|)}\right)
        \label{equ:TBR}
        \end{align}

    Parameters
    ----------
    X : tensor
        The input image, if it's complex-valued, it's amplitude is used.
    tgrs : list, optional
        target regions:[[TG1],[TG2], ..., [TGn]], [TGk] = [lefttop, rightbottom]]
    subrs : list, optional
        sub regions:[[SUB1],[SUB2], ..., [SUBn]], [SUBk] = [lefttop, rightbottom]]
    isshow : bool, optional
        show target mask given by :attr:`tgrs` (default: False)

    Returns
    -------
    TBR : float
        Target-to-Background Ratio (TBR)

    Raises
    ------
    TypeError
        tgrs mast be given
    """

    if th.is_complex(X):
        Xabs = X.abs()
    elif X.shape[-1] == 2:
        Xabs = X.pow(2).sum(-1).sqrt()
    else:
        Xabs = X.abs()

    if tgrs is None:
        raise TypeError("Please give the regions of targets!")

    if subrs is not None:
        # mask of Subregion
        SUBM = th.zeros(Xabs.shape, device=Xabs.device, dtype=th.int8)

        for subr in subrs:
            SUBM[subr[0]:subr[2], subr[1]:subr[3]] = 1
        Xabs = Xabs * SUBM

    # mask of TG
    TGM = th.zeros(Xabs.shape, device=Xabs.device, dtype=th.int8)

    for tgr in tgrs:
        TGM[tgr[0]:tgr[2], tgr[1]:tgr[3]] = 1

    # mask of BG
    BGM = 1 - TGM

    # pixel number of bgs
    NB = BGM.sum()

    R = th.max(Xabs * TGM) / (((1 / NB) * th.sum(Xabs * BGM)) + EPS)

    TBR = 20 * th.log10(R).item()

    if isshow:
        plt.figure
        plt.subplot(131)
        plt.imshow(Xabs)
        plt.subplot(132)
        plt.imshow(TGM)
        plt.subplot(133)
        plt.imshow(Xabs * TGM)
        plt.show()

    return TBR


if __name__ == '__main__':

    X = th.zeros((6, 6))
    X = th.rand((6, 6))
    # X = th.rand((6, 6)) + 1j * th.rand((6, 6))

    tgrs = [[2, 2, 4, 4]]

    X[2:4, 2:4] = 10

    print(X)

    TBR = tbr(X, tgrs=tgrs)

    print("TBR", TBR)
