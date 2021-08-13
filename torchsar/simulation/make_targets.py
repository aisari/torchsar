#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-26 09:51:56
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


from __future__ import division, print_function, absolute_import
import math
import torch as th
from ..utils.const import *
from ..dsp import normalsignals as sig
from torchsar import setseed


def gpoints(SceneArea, npoints, amax=1., seed=None):
    """Generates point targets

    Parameters
    ----------
    SceneArea : tulpe or list
        The area of scene.
    npoints : int
        The number of points.
    amax : float, optional
        The maximum amplitude.
    seed : int or None, optional
        The seed for random generator.

    Returns
    -------
    tensor
        A tensor contains coordinate and amplitude information.
    """
    if seed is not None:
        setseed(seed)
    xmin, xmax, ymin, ymax = SceneArea

    xs = th.rand(npoints, 1) * (xmax - xmin) + xmin
    ys = th.rand(npoints, 1) * (ymax - ymin) + ymin
    amps = th.rand(npoints, 1) * amax

    targets = th.zeros(npoints, 7)

    targets[:, [0]] = xs
    targets[:, [1]] = ys
    targets[:, [-1]] = amps

    return targets
