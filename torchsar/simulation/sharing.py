#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


def range_angle(sensorpos, targetpos):
    r = th.sqrt(th.sum((sensorpos - targetpos).pow(2), 0))
    return r


if __name__ == '__main__':

    pass
