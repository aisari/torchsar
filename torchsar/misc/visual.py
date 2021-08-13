#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-13 10:34:43
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import division, print_function, absolute_import
import torch as th
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchsar.utils.const import *
from torchsar.sharing.scene_and_targets import tgs2dsm
from torchsar.sharing.slant_ground_range import slantr2groundr, groundr2slantr
from skimage import exposure


def saraxis(pdict, axismod=None):

    GM = pdict['GeometryMode']
    H = pdict['H']
    V = pdict['V']
    ta = pdict['ta']
    tr = pdict['tr']
    Ar = pdict['Ar']
    if 'SceneArea' in pdict.keys():
        SceneArea = pdict['SceneArea']
    if 'SubSceneArea' in pdict.keys():
        SceneArea = pdict['SubSceneArea']
    if 'BeamArea' in pdict.keys():
        BeamArea = pdict['BeamArea']
    if 'SubBeamArea' in pdict.keys():
        BeamArea = pdict['SubBeamArea']
    if 'SceneCenter' in pdict.keys():
        SceneCenter = pdict['SceneCenter']
    if 'SubSceneCenter' in pdict.keys():
        SceneCenter = pdict['SubSceneCenter']
    if 'BeamCenter' in pdict.keys():
        BeamCenter = pdict['BeamCenter']
    if 'SubBeamCenter' in pdict.keys():
        BeamCenter = pdict['SubBeamCenter']
    if 'EchoSize' in pdict.keys():
        Na, Nr = pdict['EchoSize'][0:2]
    if 'SubEchoSize' in pdict.keys():
        Na, Nr = pdict['SubEchoSize'][0:2]
    if 'ES' in pdict.keys():
        Na, Nr = pdict['ES'][0:2]

    nTicks = 10

    if axismod is None:
        axismod = 'Image'

    if axismod == 'Image':
        extent = [1, Nr, 1, Na]
        xlabelstr = "Range"
        ylabelstr = "Azimuth"
    elif axismod == 'SceneAbsoluteSlantRange':
        if GM in ['BeamGeometry', 'BG']:
            extent = [pdict['Rnear'], pdict['Rfar'], BeamArea[2], BeamArea[3]]
        if GM in ['SceneGeometry', 'SG']:
            extent = [pdict['Rnear'], pdict['Rfar'], SceneArea[2], SceneArea[3]]
        xlabelstr = "Range (m)"
        ylabelstr = "Azimuth (m)"
    elif axismod == 'SceneRelativeSlantRange':
        if GM in ['BeamGeometry', 'BG']:
            extent = [pdict['Rnear'] - pdict['Rbc'], pdict['Rfar'] - pdict['Rbc'], BeamArea[2], BeamArea[3]]
        if GM in ['SceneGeometry', 'SG']:
            extent = [pdict['Rnear'] - pdict['Rsc'], pdict['Rfar'] - pdict['Rsc'], SceneArea[2], SceneArea[3]]
        xlabelstr = "Range (m)"
        ylabelstr = "Azimuth (m)"
    elif axismod == 'SceneAbsoluteGroundRange':
        if GM in ['BeamGeometry', 'BG']:
            X = slantr2groundr(C * tr / 2., H, Ar, 0)
            extent = [X[0], X[-1], BeamArea[2], BeamArea[3]]
        if GM in ['SceneGeometry', 'SG']:
            X = slantr2groundr(C * tr / 2., H, Ar, 0)
            extent = [X[0], X[-1], SceneArea[2], SceneArea[3]]
        # xticks = X[::int(len(X) / nTicks)]
        xlabelstr = "Range (m)"
        ylabelstr = "Azimuth (m)"
    elif axismod == 'SceneRelativeGroundRange':
        if GM in ['BeamGeometry', 'BG']:
            X = slantr2groundr(C * tr / 2., H, Ar, BeamCenter[0])
            extent = [X[0, 0], X[0, -1], BeamArea[2], BeamArea[3]]
        if GM in ['SceneGeometry', 'SG']:
            X = slantr2groundr(C * tr / 2., H, Ar, SceneCenter[0])
            extent = [X[0, 0], X[0, -1], SceneArea[2], SceneArea[3]]
        # xticks = X[::int(len(X) / nTicks)]
        xlabelstr = "Range (m)"
        ylabelstr = "Azimuth (m)"
    elif axismod == 'BeamAreaSceneArea':
        if GM in ['BeamGeometry', 'BG']:
            extent = BeamArea
        if GM in ['SceneGeometry', 'SG']:
            extent = SceneArea
        xlabelstr = "Range (m)"
        ylabelstr = "Azimuth (m)"
    else:
        extent = [ta[0, 0] * V, ta[-1, 0] * V, tr[0, 0] * C, tr[-1, 0] * C]
        xlabelstr = 'x'
        ylabelstr = 'y'
    extent = np.array(extent)
    return extent, xlabelstr, ylabelstr


def tgshow(targets, scene, shape, extent=None, cmap=None, isflip=[False, False], labelstr=None, outfile=None, isshow=True):
    r"""show targets

    Show targets in an image.

    Arguments
    --------------
    targets : Tensor
        A tensor contains information of targets.
    scene : tuple or list
        Area of scene [xmin, xmax, ymin, ymax].
    shape : tuple or list
        The shape of the scene.

    Keyword Arguments
    --------------
    outfile : str
        The filename for writting figure (default: None, do not save).
    isshow : bool
        Whether to plot figure (default: True).

    Returns
    --------------
        Tensor -- The final image tensor for show.

    Raises
    --------------
        ValueError -- :attr:`targets` should not be None.
    """

    if targets is None:
        raise ValueError("targets should not be None")

    II = tgs2dsm(targets, scene=scene, bg=0, dsize=shape, device='cpu')

    if isflip[0]:
        II = np.flipud(II)
    if isflip[1]:
        II = np.fliplr(II)

    plt.figure()
    if extent is None:
        plt.imshow(II, cmap=cmap, aspect=None, interpolation='none')
    else:
        plt.imshow(II, extent=extent, cmap=cmap, aspect=None, interpolation='none')

    if labelstr is not None:
        if labelstr[0] is not None:
            plt.xlabel(labelstr[0])
        if labelstr[1] is not None:
            plt.ylabel(labelstr[1])
        if labelstr[-1] is not None:
            plt.title(labelstr[-1])
    plt.colorbar()

    if outfile is not None:
        plt.savefig(outfile)
        print("target image has been saved to: ", outfile)
    if isshow:
        plt.show()
    plt.close()
    return II


def apshow(Srx, Title=None, cmap=None, extent=None, keepAspectRatio=True, outfile=None, isshow=True):
    r"""[summary]

    Show amplitude and phase.

    Parameters
    ----------
    Srx : Tensor
        Complex-valued tensor.
    Title : str, optional
        The figure title (the default is None, which means
        no title).
    cmap : str, optional
        The colormap (the default is None)
    extent : tulpe or list, optional
        [description] (the default is None)
    keepAspectRatio : bool, optional
        [description] (the default is True)
    outfile : str, optional
        The filename for outputing (the default is None, which means not save)
    isshow : bool, optional
        Whether to show the figure (the default is True).
    """
    if Title is None:
        Title = 'SAR raw data'

    plt.figure()
    plt.subplot(121)
    plt.imshow(np.absolute(
        # Srx), aspect='auto' if not keepAspectRatio else None,
        # interpolation='none')
        Srx), extent=extent, aspect='auto' if not keepAspectRatio else None, interpolation='none', cmap=cmap)
    # plt.xlabel("Range (m)")
    # plt.ylabel("Azimuth (m)")
    plt.xlabel("Range\n(a)")
    plt.ylabel("Azimuth")
    plt.title(Title + " (amplitude)")

    plt.subplot(122)
    # plt.xlabel("Range (m)")
    # plt.ylabel("Azimuth (m)")
    plt.xlabel("Range\n(b)")
    plt.ylabel("Azimuth")
    plt.imshow(np.angle(Srx), extent=extent,
               aspect='auto' if not keepAspectRatio else None, interpolation='none', cmap=cmap)
    plt.title(Title + " (phase)")
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
        print("image has been saved to: ", outfile)
    # plt.colorbar()
    if isshow:
        plt.show()
    else:
        plt.close()


def show_response(Srx, extent=None, title="Figure", keepAspectRatio=True, outfile=None, isshow=True):
    r"""show SAR response

    [description]

    Arguments
    ------------------
    Srx : Tensor
        [description]

    Keyword Arguments
    ------------------
    extent : tuple, list or None
        [description] (default: None)
    title : str
        [description] (default: "Figure")
    keepAspectRatio : bool
        [description] (default: True)
    outfile : str
        [description] (default: None)
    isshow : bool
        [description] (default: True)
    """

    plt.figure()
    plt.title(title)
    plt.xlabel("Range (m)")
    plt.ylabel("Azimuth (m)")
    extent = [27844.409098089047, 28868.409098089047, 512.0, -512.0]
    plt.imshow(np.absolute(Srx), extent=extent,
               aspect='auto' if not keepAspectRatio else None, interpolation='none')
    plt.colorbar()
    if outfile is not None:
        plt.savefig(outfile)
        print("image has been saved to: ", outfile)
    if isshow:
        plt.show()
    else:
        plt.close()


def showReImAmplitudePhase(Srx, extent=None, title="Figure", keepAspectRatio=True):
    r"""[summary]

    [description]

    Parameters
    ----------
    Srx : {[type]}
        [description]
    extent : {[type]}, optional
        [description] (the default is None, which [default_description])
    title : str, optional
        [description] (the default is "Figure", which [default_description])
    keepAspectRatio : bool, optional
        [description] (the default is True, which [default_description])
    """

    f = plt.figure()
    f.suptitle(title)
    plt.subplot(221)
    plt.title("SAR response, module")
    # plt.xlabel("Range (m)")
    # plt.ylabel("Azimuth (m)")
    plt.xlabel("Range")
    plt.ylabel("Azimuth")
    plt.imshow(np.absolute(Srx), extent=extent,
               aspect='auto' if not keepAspectRatio else None, interpolation='none')
    plt.colorbar()
    plt.subplot(222)
    plt.title("SAR response (phase)")
    plt.xlabel("Range")
    plt.ylabel("Azimuth")
    plt.imshow(np.angle(Srx), extent=extent,
               aspect='auto' if not keepAspectRatio else None, interpolation='none')
    plt.colorbar()
    plt.subplot(223)
    plt.title("SAR response (real part)")
    plt.xlabel("Range")
    plt.ylabel("Azimuth")
    plt.imshow(np.real(Srx), extent=extent,
               aspect='auto' if not keepAspectRatio else None, interpolation='none')
    plt.colorbar()
    plt.subplot(224)
    plt.title("SAR response (imaginary part)")
    plt.xlabel("Range")
    plt.ylabel("Azimuth")
    plt.imshow(np.imag(Srx), extent=extent,
               aspect='auto' if not keepAspectRatio else None, interpolation='none')
    plt.colorbar()
    plt.show()


def show_image(img, Title=None, cmap=None, keepAspectRatio=True, outfile=None, isshow=True):
    H = img.shape[0]
    W = img.shape[1]

    if Title is None:
        # Title = "Image"
        Title = "Intensity"
    extent = [-W / 2.0, W / 2.0, -H / 2.0, H / 2.0]
    # extent = [0, W, H, 0]

    plt.figure()
    ax = plt.subplot(111)
    plt.title(Title)
    plt.xlabel("Range")
    plt.ylabel("Azimuth")
    # plt.xticks(fontsize=17)
    # plt.yticks(fontsize=17)
    # ax.set_xlabel("Range", fontsize=17)
    # ax.set_ylabel("Azimuth", fontsize=17)
    # ax.legend(fontsize=17)
    plt.imshow(img, extent=extent,
               aspect='auto' if not keepAspectRatio else None, interpolation='none', cmap=cmap)

    plt.colorbar()

    if outfile is not None:
        plt.savefig(outfile)
        print("image has been saved to: ", outfile)

    if isshow:
        plt.show()
        pass
    else:
        plt.close()
    return img


def sarshow(SI, pdict, axismod=None, title=None, cmap=None, aspect=None, outfile=None, newfig=True, figsize=None):
    r"""show sar image

    show sar image

    Arguments:
        SI {[type]} -- [description]
        pdict {[type]} -- [description]

    Keyword Arguments:
        axismod {[type]} -- [description] (default: {None})
        title {[type]} -- [description] (default: {None})
        cmap {[type]} -- [description] (default: {None})
        aspect {[type]} -- [description] (default: {None})
        outfile {[type]} -- [description] (default: {None})
        newfig bool -- [description] (default: {True})
        figsize {[type]} -- [description] (default: {None})
    """

    extent, xlabelstr, ylabelstr = saraxis(pdict=pdict, axismod=axismod)

    if newfig:
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

    if type(SI) is th.Tensor:
        SI = SI.cpu().numpy()
    if axismod in ['image', 'Image', 'IMAGE']:
        ax.imshow(SI, aspect=aspect, interpolation='none', cmap=cmap)
    else:
        ax.imshow(SI, extent=extent, aspect=aspect, interpolation='none', cmap=cmap)
    plt.xlabel(xlabelstr)
    plt.ylabel(ylabelstr)
    plt.title(title)

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
        print("sar image has been saved to: ", outfile)

    if newfig:
        plt.show()


def show_sarimage(SI, pdict, axismod=None, title=None, cmap=None, isimgadj=False, aspect=None, outfile=None, newfig=True, figsize=None):
    r"""[summary]

    [description]

    Arguments:
        SI {[type]} -- [description]
        pdict {[type]} -- [description]

    Keyword Arguments:
        axismod {[type]} -- [description] (default: {None})
        title {[type]} -- [description] (default: {None})
        cmap {[type]} -- [description] (default: {None})
        isimgadj bool -- [description] (default: {False})
        aspect {[type]} -- [description] (default: {None})
        outfile {[type]} -- [description] (default: {None})
        newfig bool -- [description] (default: {True})
        figsize {[type]} -- [description] (default: {None})
    """

    extent, xlabelstr, ylabelstr = saraxis(pdict=pdict, axismod=axismod)

    if newfig:
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

    plt.xlabel(xlabelstr)
    plt.ylabel(ylabelstr)

    SI = np.flipud(SI)

    A = np.absolute(SI)

    plt.title(title)

    if isimgadj:
        A = A / A.max()
        A = 20 * np.log10(A + EPS)
        a = np.abs(A.mean())
        A = (A + a) / a * 255
        A[A < 0] = 0
        A = A.astype('uint8')
        A = np.flipud(A)

    if axismod in ['image', 'Image', 'IMAGE']:
        ax.imshow(A, aspect=aspect, interpolation='none', cmap=cmap)
    else:
        ax.imshow(A, extent=extent, aspect=aspect, interpolation='none', cmap=cmap)

    if axismod == 'SceneAbsoluteGroundRange' or axismod == 'SceneRelativeGroundRange':
        # plt.xticks(xticks)
        pass
        # ax.xaxis.set_major_locator(FixedLocator(X[::int(len(X) / 10.)]))
    # plt.colorbar()
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
        print("sar image has been saved to: ", outfile)

    if newfig:
        plt.show()


def show_sarimage3d(SI, pdict, axismod=None, title=None, cmap=None, isimgadj=False, aspect=None, outfile=None, figsize=None):
    r"""[summary]

    [description]

    Arguments:
        SI {[type]} -- [description]
        sarplat {[type]} -- [description]

    Keyword Arguments:
        axismod {[type]} -- [description] (default: {None})
        title {[type]} -- [description] (default: {None})
        cmap {[type]} -- [description] (default: {None})
        isimgadj bool -- [description] (default: {False})
        aspect {[type]} -- [description] (default: {None})
        outfile {[type]} -- [description] (default: {None})
        figsize {[type]} -- [description] (default: {None})
    """

    extent, xlabelstr, ylabelstr = saraxis(pdict=pdict, axismod=axismod)

    Z = np.absolute(SI)
    if isimgadj:
        Z = exposure.adjust_log(Z)
    Z = np.flipud(Z)
    # fig = plt.figure(figsize=figsize)
    M, N = np.shape(Z)

    X = np.arange(0, N, 1)
    Y = np.arange(0, M, 1)
    X, Y = np.meshgrid(X, Y)

    # mlab.figure(size=(400, 500))
    # mlab.mesh(X, Y, Z)
    # mlab.surf(X, Y, Z)
    # mlab.colorbar()
    # mlab.xlabel(xlabelstr)
    # mlab.ylabel(ylabelstr)
    # mlab.zlabel("Amplitude")
    # mlab.title(title)
    # mlab.show()

    if outfile is not None:
        # mlab.savefig(outfile)
        print("sar image has been saved to: ", outfile)


def imshow(
        X, cmap=None, norm=None, aspect=None, interpolation=None,
        alpha=None, vmin=None, vmax=None, origin=None, extent=None,
        filternorm=True, filterrad=4.0, resample=None, url=None,
        data=None, **kwargs):
    """show an image

    This function create an figure and show an image, see ``pyplot.imshow`` for documentation.

    """

    if type(X) is th.Tensor:
        X = X.cpu().numpy()
    plt.figure()
    plt.imshow(
        X, cmap=cmap, norm=norm, aspect=aspect, interpolation=interpolation,
        alpha=alpha, vmin=vmin, vmax=vmax, origin=origin, extent=extent,
        filternorm=True, filterrad=4.0, resample=None, url=None,
        data=None, **kwargs)
    plt.show()
    return 0
