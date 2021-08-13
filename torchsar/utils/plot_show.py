#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def cplot(ca, lmod=None):
    N = len(ca)
    if lmod is None:
        lmod = '-b'
    r = ca.real
    i = ca.imag
    # x = np.hstack((np.zeros(N), r))
    # y = np.hstack((np.zeros(N), i))
    for n in range(N):
        plt.plot([0, r[n]], [0, i[n]], lmod)
    plt.xlabel('real')
    plt.ylabel('imag')


def plots(x, ydict, plotdir='./', xlabel='x', ylabel='y', title='', issave=False, isshow=True):
    if type(x) is th.Tensor:
        x = x.detach().cpu().numpy()
    legend = []
    plt.figure()
    plt.grid()
    for k, v in ydict.items():
        if type(v) is th.Tensor:
            v = v.detach().cpu().numpy()
        plt.plot(x, v)
        legend.append(k)
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if issave:
        plt.savefig(plotdir + ylabel + '_' + xlabel + '.png')
    if isshow:
        plt.show()
    plt.close()


class Plots:

    def __init__(self, plotdir='./', xlabel='x', ylabel='y', title='', figname=None, issave=False, isshow=True):

        self.plotdir = plotdir
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.issave = issave
        self.isshow = isshow
        if figname is None or figname == '':
            self.figname = self.plotdir + self.ylabel + '_' + self.xlabel + '.png'
        else:
            self.figname = figname

    def __call__(self, x, ydict, figname=None):

        if figname is None or figname == '':
            figname = self.figname

        if type(x) is th.Tensor:
            x = x.detach().cpu().numpy()
        legend = []
        plt.figure()
        plt.grid()
        for k, v in ydict.items():
            if type(v) is th.Tensor:
                v = v.detach().cpu().numpy()
            plt.plot(x, v)
            legend.append(k)
        plt.legend(legend)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        if self.issave:
            plt.savefig(figname)
        if self.isshow:
            plt.show()
        plt.close()



if __name__ == '__main__':

    N = 3

    r = np.random.rand(N)
    i = -np.random.rand(N)

    print(r)
    print(i)
    x = r + 1j * i

    cplot(x)
    plt.show()

    Ns = 100
    x = th.linspace(-1, 1, Ns)

    y = th.randn(Ns)
    f = th.randn(Ns)

    plot = Plots(plotdir='./', issave=True)
    plot(x, {'y': y, 'f': f})

