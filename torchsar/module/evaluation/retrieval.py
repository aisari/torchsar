#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


import torch as th
import torch.nn as nn
from torchsar.utils.const import EPS
from torch.autograd import Variable


class Dice(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(Dice, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def soft_dice_coeff(self, P, G):

        cupnum = th.sum(P, (1, 2, 3)) + th.sum(G, (1, 2, 3))
        capnum = th.sum(P * G, (1, 2, 3))

        score = (2. * capnum + EPS) / (cupnum + EPS)
        if self.reduce is True:
            if self.size_average is True:
                score = score.mean()
            else:
                score = score.sum()

        return score

    def __call__(self, P, G):
        return self.soft_dice_coeff(P, G)


class Jaccard(nn.Module):
    r"""Jaccard distance

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

    def __init__(self, size_average=True, reduce=True):
        super(Jaccard, self).__init__()

    def forward(self, P, G):
        capnum = th.sum(P * G)
        cupnum = th.sum(P + G)

        return capnum / (cupnum - capnum + EPS)


class Iridescent(nn.Module):
    r"""Iridescent Distance

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

    def __init__(self, size_average=True, reduce=True):
        super(Iridescent, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, P, G):
        sumbothones = th.sum(P * G)
        sumbothzeros = th.sum((1 - P) * (1 - G))
        print(sumbothones, sumbothzeros)
        return (sumbothones + sumbothzeros) / (th.sum(P) + th.sum(G) - sumbothones + sumbothzeros + EPS)


class F1(nn.Module):
    r"""F1 distance

    .. math::
       F_{\beta} = 1 -\frac{(1+\beta^2) * P * R}{\beta^2 *P + R}
       :label: equ-F-measure

    where,

    .. math::
       {\rm PPV} = {P} = \frac{\rm TP}{{\rm TP} + {\rm FP}}
       :label: equ-Precision

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

    """

    def __init__(self, size_average=True, reduce=True):
        super(F1, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, P, G):

        TP = th.sum(P * G)
        FP = th.sum(P * (1. - G))
        FN = th.sum((1 - P) * G)
        Precision = TP / (TP + FP + EPS)
        Recall = TP / (TP + FN + EPS)

        return Precision * Recall / (Precision + Recall + EPS)


if __name__ == '__main__':
    import numpy as np

    # X = np.array([[1, 1, 1], [0, 1, 0]])
    # X = np.array([[0, 0, 1], [0, 0, 1]])
    X = np.array([[0.3, 0, 1], [0, 0.8, 1]])
    X = np.array([[0, 0, 1], [0, 0, 1]])
    G = np.array([[0, 0, 1], [0, 0, 1]])

    X = np.array([[0.1, 0.1, 0], [0, 0, 0]])
    G = np.array([[0, 0.5, 0], [0, 0, 0]])

    # X = np.array([[1, 1, 0], [0, 0, 0]])
    # G = np.array([[1, 1, 1], [1, 1, 1]])

    # X = np.array([[1, 1, 1], [1, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 0, 0]])

    # X = np.array([[0, 1, 0], [0, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 1, 0]])

    # X = np.array([[0, 0, 0], [0, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 0, 0]])

    # X = np.array([[1, 0, 1], [1, 1, 1]])
    # G = np.array([[1, 1, 1], [1, 1, 1]])

    X = np.array([[1, 1, 1], [1, 1, 1]])
    G = np.array([[1, 1, 1], [1, 1, 1]])

    X = th.randn(4, 1, 3, 2)
    G = th.randn(4, 1, 3, 2)

    X = th.Tensor(X)
    G = th.Tensor(G)
    X = Variable(X, requires_grad=True)
    G = Variable(G, requires_grad=True)

    net = nn.ReLU()
    P = net(X)

    print(X)
    print(P)
    print(G)
    print(th.mean(th.abs(X - G)))
