#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 22:29:14
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

import torch as th
from torchsar.base.arrayops import sl


def draw_rectangle(x, rects, edgecolors=[[255, 0, 0]], linewidths=[1], fillcolors=[None], axes=(-3, -2)):
    """Draw rectangles in a tensor


    Parameters
    ----------
    x : tensor
        The input with any size.
    rects : list or tuple
        The coordinates of the rectangles [[lefttop, rightbottom]].
    edgecolors : list, optional
        The color of edge.
    linewidths : int, optional
        The linewidths of edge.
    fillcolors : int, optional
        The color for filling.
    axes : int, optional
        The axes for drawing the rect (default [(-3, -2)]).
    """

    axes = axes * len(rects) if len(axes) == 1 and len(rects) > 1 else axes

    if type(x) is not th.Tensor:
        x = th.tensor(x)
    d = x.dim()

    for rect, edgecolor, linewidth, fillcolor, axis in zip(rects, edgecolors, linewidths, fillcolors, axes):
        edgecolor = th.tensor(edgecolor, dtype=x.dtype) if edgecolor is not None else None
        fillcolor = th.tensor(fillcolor, dtype=x.dtype) if fillcolor is not None else None
        if edgecolor is not None:
            top, left, bottom, right = rect
            for l in range(linewidth):
                x[sl(d, axis, [slice(top, bottom + 1), [left, right]])] = edgecolor
                x[sl(d, axis, [[top, bottom], slice(left, right)])] = edgecolor
                top += 1
                left += 1
                bottom -= 1
                right -= 1
        if fillcolor is not None:
            top, left, bottom, right = rect
            top += linewidth
            left += linewidth
            bottom -= linewidth
            right -= linewidth
            x[sl(d, axis, [slice(top, bottom + 1), slice(left, right + 1)])] = fillcolor
    return x


def draw_eclipse(x, centroids, aradii, bradii, edgecolors=[255, 0, 0], linewidths=1, fillcolors=None, axes=(-2, -1)):

    for centroid, aradius, bradius in centroids, aradii, bradii:
        pass


if __name__ == '__main__':

    x = th.zeros(2, 8, 10, 3)

    rects = [[1, 2, 6, 8]]

    y = draw_rectangle(x, rects, edgecolors=[[100, 125, 255]], linewidths=[2], fillcolors=[None], axes=[(-3, -2)])

    print(x[0, :, :, 0])
    print(x[0, :, :, 1])
    print(x[0, :, :, 2])

    print(y[0, :, :, 0])
    print(y[0, :, :, 1])
    print(y[0, :, :, 2])

    y = draw_rectangle(x, rects, edgecolors=[[100, 125, 255]], linewidths=[2], fillcolors=[[20, 50, 80]], axes=[(-3, -2)])

    print(x[0, :, :, 0])
    print(x[0, :, :, 1])
    print(x[0, :, :, 2])

    print(y[0, :, :, 0])
    print(y[0, :, :, 1])
    print(y[0, :, :, 2])
