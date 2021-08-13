#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th
from torchsar.utils.const import EPS
from torchsar.base.mathops import nextpow2
from torchsar.base.arrayops import sl


def standardization(X, mean=None, std=None, axis=None, extra=False):
    r"""standardization

    .. math::
        \bar{X} = \frac{X-\mu}{\sigma}


    Parameters
    ----------
    X : Tensor
        data to be normalized,
    mean : list or None, optional
        mean value (the default is None, which means auto computed)
    std : list or None, optional
        standard deviation (the default is None, which means auto computed)
    axis : list or int, optional
        specify the axis for computing mean and standard deviation (the default is None, which means all elements)
    extra : bool, optional
        if True, also return the mean and std (the default is False, which means just return the standardized data)
    """

    if type(X) is np.tensor:
        X = th.from_numpy(X)

    if mean is None:
        if axis is None:
            mean = th.mean(X)
        else:
            mean = th.mean(X, axis, keepdim=True)
    if std is None:
        if axis is None:
            std = th.std(X)
        else:
            std = th.std(X, axis, keepdim=True)
    if extra is True:
        return (X - mean) / (std + EPS), mean, std
    else:
        return (X - mean) / (std + EPS)


def scale(X, st=[0, 1], sf=None, istrunc=True, extra=False):
    r"""
    Scale data.

    .. math::
        x \in [a, b] \rightarrow y \in [c, d]

    .. math::
        y = (d-c)*(x-a) / (b-a) + c.

    Parameters
    ----------
    X : tensor_like
        The data to be scaled.
    st : tuple, list, optional
        Specifies the range of data after beening scaled. Default [0, 1].
    sf : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    istrunc : bool
        Specifies wether to truncate the data to [a, b], For example,
        If sf == [a, b] and 'istrunc' is true,
        then X[X < a] == a and X[X > b] == b.
    extra : bool
        If ``True``, also return :attr:`st` and :attr:`sf`.

    Returns
    -------
    out : tensor
        Scaled data tensor.
    st, sf : list or tuple
        If :attr:`extra` is true, also be returned
    """

    if type(X) is np.tensor:
        X = th.from_numpy(X)

    X = X.float()

    if not(isinstance(st, (tuple, list)) and len(st) == 2):
        raise Exception("'st' is a tuple or list, such as (-1,1)")
    if sf is not None:
        if not(isinstance(sf, (tuple, list)) and len(sf) == 2):
            raise Exception("'sf' is a tuple or list, such as (0, 255)")
    else:
        sf = [th.min(X) + 0.0, th.max(X) + 0.0]
    if sf[0] is None:
        sf = (th.min(X) + 0.0, th[1])
    if sf[1] is None:
        sf = (sf[0], th.max(X) + 0.0)

    a = sf[0] + 0.0
    b = sf[1] + 0.0
    c = st[0] + 0.0
    d = st[1] + 0.0

    if istrunc:
        X[X < a] = a
        X[X > b] = b

    if extra:
        return (X - a) * (d - c) / (b - a + EPS) + c, st, sf
    else:
        return (X - a) * (d - c) / (b - a + EPS) + c


def quantization(X, idrange=None, odrange=[0, 31], odtype='auto', extra=False):
    r"""
    Quantize data.

    .. math::
        x \in [a, b] \rightarrow y \in [c, d]

    .. math::
        y = (d-c)*(x-a) / (b-a) + c.

    Parameters
    ----------
    X : tensor
        The data to be quantized with shape :math:`N_a×N_r ∈ {\mathbb R}`, or :math:`N_a×N_r ∈ {\mathbb C}`.
    idrange : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    odrange : tuple, list, optional
        Specifies the range of data after beening quantized. Default [0, 31].
    odtype : str, None, optional
        output data type, supportted are ``'auto'`` (auto infer, default), or torch tensor's dtype string.
        If the type of :attr:`odtype` is not string(such as None),
        the type of output data is the same with input.
    extra : bool
        If ``True``, also return :attr:`st` and :attr:`idrange`.

    Returns
    -------
    out : tensor
        Quantized data tensor, if the input is complex, will return a tensor with shape :math:`N_a×N_r×2 ∈ {\mathbb R}`.
    idrange, odrange : list or tuple
        If :attr:`extra` is true, also be returned
    """

    if type(X) is np.tensor:
        X = th.from_numpy(X)

    if th.is_complex(X):
        X = th.view_as_real(X)

    if not(isinstance(odrange, (tuple, list)) and len(odrange) == 2):
        raise Exception("'st' is a tuple or list, such as (-1,1)")
    if idrange is not None:
        if not(isinstance(idrange, (tuple, list)) and len(idrange) == 2):
            raise Exception("'sf' is a tuple or list, such as (0, 255)")
    else:
        idrange = [X.min() + 0.0, X.max() + 0.0]
    if idrange[0] is None:
        idrange = (X.min() + 0.0, idrange[1])
    if idrange[1] is None:
        idrange = (idrange[0], X.max() + 0.0)

    a = idrange[0] + 0.0
    b = idrange[1] + 0.0
    c = odrange[0] + 0.0
    d = odrange[1] + 0.0

    X[X < a] = a
    X[X > b] = b

    X = (X - a) * (d - c) / (b - a + EPS) + c

    if odtype in ['auto', 'AUTO']:
        if odrange[0] >= 0:
            odtype = 'th.uint'
        else:
            odtype = 'th.int'
        odtype = odtype + str(nextpow2(odrange[1] - odrange[0]))

    if type(odtype) is str:
        X = X.to(eval(odtype))

    if extra:
        return X, idrange, odrange
    else:
        return X


def db20(x):
    """Computes dB value of a tensor

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor (dB)
    """
    return 20. * th.log10(th.abs(x))


def ct2rt(x, axis=0):
    r"""Converts a complex-valued tensor to a real-valued tensor

    Converts a complex-valued tensor :math:`{\bf x}` to a real-valued tensor with FFT and conjugate symmetry.


    Parameters
    ----------
    x : Tensor
        The input tensor :math:`{\bf x}\in {\mathbb C}^{H×W}`.
    axis : int
        The axis for excuting FFT.

    Returns
    -------
    Tensor
        The output tensor :math:`{\bf y}\in {\mathbb R}^{2H×W}` ( :attr:`axis` = 0 ), :math:`{\bf y}\in {\mathbb R}^{H×2W}` ( :attr:`axis` = 1 )
    """

    d = x.dim()
    n = x.shape[axis]
    X = th.fft.fft(x, axis=axis)
    X0 = X[sl(d, axis, [[0]])]
    X1 = th.conj(X[sl(d, axis, range(n - 1, 0, -1))])
    Y = th.cat((X, X0.imag, X1), dim=axis)
    Y[sl(d, axis, [[0]])] = X0.real + 0j
    del x, X, X1
    y = th.fft.ifft(Y, axis=axis)
    return y


def rt2ct(y, axis=0):
    r"""Converts a real-valued tensor to a complex-valued tensor

    Converts a real-valued tensor :math:`{\bf y}` to a complex-valued tensor with FFT and conjugate symmetry.


    Parameters
    ----------
    y : Tensor
        The input tensor :math:`{\bf y}\in {\mathbb C}^{2H×W}`.
    axis : int
        The axis for excuting FFT.

    Returns
    -------
    Tensor
        The output tensor :math:`{\bf x}\in {\mathbb R}^{H×W}` ( :attr:`axis` = 0 ), :math:`{\bf x}\in {\mathbb R}^{H×W}` ( :attr:`axis` = 1 )
    """

    d = y.dim()
    n = y.shape[axis]

    Y = th.fft.fft(y, axis=axis)
    X = Y[sl(d, axis, range(0, int(n / 2)))]
    X[sl(d, axis, [[0]])].imag = Y[sl(d, axis, [[int(n / 2)]])].real
    del y, Y
    x = th.fft.ifft(X, axis=axis)
    return x


if __name__ == '__main__':

    X = th.randn(4, 3, 5, 6)
    # X = th.randn(3, 4)
    XX = standardization(X, axis=(0, 2, 3))
    XX, meanv, stdv = standardization(X, axis=(0, 2, 3), extra=True)
    print(XX.size())
    print(meanv, stdv)

    X = np.random.randn(4, 3, 5, 6) * 255
    # X = th.randn(3, 4)
    XX = standardization(X, axis=(0, 2, 3))
    XX, meanv, stdv = standardization(X, axis=(0, 2, 3), extra=True)
    print(XX.size())
    print(meanv, stdv)
    print(XX)

    XX = scale(X, st=[0, 1])
    print(XX)

    x = th.randn(8, 8) + 1j * th.randn(8, 8)
    y = ct2rt(x, axis=0)
    print(x, x.shape)
    print(y, y.shape)
    z = rt2ct(y, axis=0)
    print(z, z.shape)

    print(y.imag.sum())
    print((x - z).abs().sum())
