#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-02-18 11:06:13
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


def polyfit(x, y, deg=1, C=1):
    r"""Least squares polynomial fit.

    We fit the data using a polynomial function of the form

    .. math::
        y(x, \mathbf{w})=w_{0}+w_{1} x+w_{2} x^{2}+, \cdots,+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}

    where :math:`M` is the order of the polynomial, and :math:`x^{j}` denotes :math:`x` raised to
    the power of :math:`j`, The polynomial coefficients :math:`w_{0}, w_{1}, \cdots \cdot w_{M}`
    comprise the vector :math:`\mathbf{w}=\left(w_{0}, w_{1}, \cdots \cdot w_{M}\right)^{\mathrm{T}}`.

    This can be done by minimizing an error function

    ..math::
        E(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2} + C \|{\mathbf w}\|_2^2

    the solution is

    .. math::
        \mathbf{w}=\left(\mathbf{X}^{\mathrm{T}} \mathbf{X}\right)^{-1} \mathbf{X}^{\mathrm{T}} \mathbf{t}


    see https://iridescent.blog.csdn.net/article/details/39554293

    Parameters
    ----------
    x : {torch 1d tensor}
        x-coordinates of the N sample points :math:`(x[i], y[i])`
    y : {torch 1d tensor}
        y-coordinates of the N sample points :math:`(x[i], y[i])`
    deg : {number or tuple or list}, optional
        degree (:math:`M`) of the fitting polynomial, ``deg[0]`` is the minimum degree,
        ``deg[1]`` is the maximum degree, if ``deg`` is an integer, the minimum degree is 0
        , the maximum degree is deg (the default is 1)
    C : float, optional
        the balance factor of weight regularization and fitting error.

    Returns
    ----------
    w : {torch 1d tensor}
        The polynomial coefficients :math:`w_{0}, w_{1}, \cdots \cdot w_{M}`.

    """

    if type(deg) is int:
        deg = (0, deg)

    if x.dim() == 1 or (x.dim() == 2 and (x.size(0) == 1 or x.size(1) == 1)):
        x = x.view(1, -1)
    if y.dim() == 1 or (y.dim() == 2 and (y.size(0) == 1 or y.size(1) == 1)):
        y = y.view(1, -1)

    xs = th.tensor([], device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
    for m in range(deg[0], deg[1] + 1):
        xs = th.cat((xs, x**m), axis=0)
    II = th.eye(deg[1] - deg[0] + 1, deg[1] - deg[0] + 1) / C
    II = II.to(x.device)
    w = th.inverse(th.matmul(xs, xs.t()) + II).mm(xs).mm(y.t())
    return w


def polyval(w, x, deg=None):
    r"""Evaluate a polynomial at specific values.

    We fit the data using a polynomial function of the form

    .. math::
        y(x, \mathbf{w})=w_{0}+w_{1} x+w_{2} x^{2}+, \cdots,+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}

    Parameters
    ----------
    w : tensor
        the polynomial coefficients :math:`w_{0}, w_{1}, \cdots \cdot w_{M}` with shape M-N, where N is the number of coefficients, and M is the degree.
    x : 1d tensor
        x-coordinates of the N sample points :math:`(x[i], y[i])`
    deg : int or tuple or list or None, optional
        degree (:math:`M`) of the fitting polynomial, ``deg[0]`` is the minimum degree,
        ``deg[1]`` is the maximum degree; if ``deg`` is an integer, the minimum degree is 0,
        the maximum degree is ``deg``; if ``deg`` is ``None``, the minimum degree is 0
        the maximum degree is ``w.shape[1] - 1``(the default is None)

    Returns
    ----------
    y : 1d tensor
        y-coordinates of the N sample points :math:`(x[i], y[i])`

    """

    if deg is None:
        deg = (0, w.shape[0] - 1)
    if type(deg) is int:
        deg = (0, deg)

    if x.dim() == 1 or (x.dim() == 2 and (x.size(0) == 1 or x.size(1) == 1)):
        x = x.view(1, -1)

    xs = th.tensor([], device=x.device, dtype=x.dtype)
    for m in range(deg[0], deg[1] + 1):
        xs = th.cat((xs, x**m), axis=0)
    y = th.matmul(xs.t(), w)

    return y


def rmlinear(x, y, deg=2, C=1):
    r"""Remove linear trend

    After the data is fitted by

    .. math::
        y(x, \mathbf{w})=w_{0}+w_{1} x+w_{2} x^{2}+, \cdots,+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}

    the constant and linear trend can be removed by setting :math:`w_{0}=w_{1}=0`, or simply by fitting
    the data with 2 to :math:`M` order polynominal.

    Parameters
    ----------
    x : 1d tensor
        x-coordinates of the N sample points :math:`(x[i], y[i])`
    y : 1d tensor
        y-coordinates of the N sample points :math:`(x[i], y[i])`
    deg : int, optional
        degree (:math:`M`) of the fitting polynomial (the default is 1)
    C : float, optional
        the balance factor of weight regularization and fitting error.

    Returns
    -------
    y : 1d tensor
        y-coordinates of the N sample points :math:`(x[i], y[i])`, linear trend is removed.
    """

    w = polyfit(x, y, deg=deg, C=C)
    w[0, :] = 0.
    w[1, :] = 0.
    y = polyval(w, x)
    return y.t()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    Ns, k, b = 100, 6.2, 3.0
    x = th.linspace(0, 1, Ns)
    y = x * k + b + th.randn(Ns)
    # x, y = x.to('cuda:1'), y.to('cuda:1')

    w = polyfit(x, y, deg=1)
    print(w, w.shape)
    w = polyfit(x, y, deg=1, C=5)
    print(w)
    yy = polyval(w, x)
    print(yy)

    print(th.sum(th.abs(y - yy)))

    plt.figure()
    plt.plot(x, y, 'ob')
    plt.plot(x, yy, '-r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('polynomial fitting')
    plt.legend(['noised data', 'fitted'])
    plt.show()
