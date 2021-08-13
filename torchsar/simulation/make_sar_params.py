#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import print_function
import math
import torch as th
import torchsar as ts


class SARParameterGenerator(object):

    """SAR Parameter Generator

    Attributes
    ----------
    prdict : dict
        ``{'p1': [low, high], 'p2': [low, high]...}``
    seed : int or None
        The seed for random generator.
    """

    def __init__(self, prdict, seed=None):
        self.prdict = prdict
        self.seed = seed
        if seed is not None:
            ts.setseed(seed)

    def mksarp(self, n=1, seed=None):
        """Makes SAR Parameters

        Parameters
        ----------
        n : int, optional
            The number of experiments.
        seed : None, optional
            The seed for random generator.

        Returns
        -------
        dict
            The SAR Parameter dict.
        """
        if seed is not None:
            ts.setseed(seed)

        pdict = {}
        for k, v in self.prdict.items():
            if v is not None:
                low, high = v
                if n == 1:
                    pdict[k] = th.rand(n).item() * (high - low) + low
                else:
                    pdict[k] = th.rand(n) * (high - low) + low

        return pdict


if __name__ == '__main__':

    seed = 2020
    prdict = {}
    prdict['H'] = (0, 100)
    prdict['V'] = (11, 22)

    sarg = SARParameterGenerator(prdict, seed=seed)

    pdict = sarg.mksarp()

    print(pdict)
