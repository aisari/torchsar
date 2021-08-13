#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torchsar as ts


def antenna_pattern_azimuth(Wl, La, A):

    BWa = 0.886 * Wl / La

    Pa = ts.sinc(0.886 * A / BWa)

    return Pa


if __name__ == '__main__':

    Pa = antenna_pattern_azimuth(0.25, 2, 0.2)
    print(Pa)
