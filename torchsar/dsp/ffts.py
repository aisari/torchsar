#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th
import torch.fft as thfft


def freq(n, fs, norm=False, shift=False, dtype=th.float32, device='cpu'):
    r"""Return the sample frequencies

    Return the sample frequencies.

    Given a window length `n` and a sample rate `fs`, if shift is ``True``::

      f = [-n/2, ..., n/2] / (d*n)

    Given a window length `n` and a sample rate `fs`, if shift is ``False``::

      f = [0, 1, ..., n] / (d*n)

    If :attr:`norm` is ``True``, :math:`d = 1`, else :math:`d = 1/f_s`.

    Parameters
    ----------
    fs : float
        Sampling rate.
    n : int
        Number of samples.
    norm : bool
        Normalize the frequencies.
    shift : bool
        Does shift the zero frequency to center.
    dtype : Tensor
        Data type, default is ``th.float32``.
    device : str
        device string, default is ``'cpu'``.

    Returns
    -------
    torch 1d-tensor
        Frequency array with size :math:`n×1`.
    """

    d = 1. / fs

    if shift:
        f = np.linspace(-n / 2., n / 2., n, endpoint=True)
    else:
        f = np.linspace(0, n, n, endpoint=True)

    if norm:
        return th.tensor(f / n, dtype=dtype, device=device)
    else:
        return th.tensor(f / (d * n), dtype=dtype, device=device)


def fftfreq(n, fs, norm=False, shift=False, dtype=th.float32, device='cpu'):
    r"""Return the Discrete Fourier Transform sample frequencies

    Return the Discrete Fourier Transform sample frequencies.

    Given a window length `n` and a sample rate `fs`, if shift is ``True``::

      f = [-n/2, ..., -1,     0, 1, ...,   n/2-1] / (d*n)   if n is even
      f = [-(n-1)/2, ..., -1, 0, 1, ..., (n-1)/2] / (d*n)   if n is odd

    Given a window length `n` and a sample rate `fs`, if shift is ``False``::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    where :math:`d = 1/f_s`, if :attr:`norm` is ``True``, :math:`d = 1`, else :math:`d = 1/f_s`.

    Parameters
    ----------
    n : int
        Number of samples.
    fs : float
        Sampling rate.
    norm : bool
        Normalize the frequencies.
    shift : bool
        Does shift the zero frequency to center.
    dtype : {torch tensor type}
        Data type, default is ``th.float32``.
    dtype : {torch tensor type}
        device string, ``'cpu', 'cuda:0', 'cuda:1'``

    Returns
    -------
    torch 1d-array
        Frequency array with size :math:`n×1`.
    """

    # d = 1. / fs
    # if n % 2 == 0:
    #     N = n
    #     N1 = int(n / 2.)
    #     N2 = int(n / 2.)
    #     endpoint = False
    # else:
    #     N = n - 1
    #     N1 = int((n + 1) / 2.)
    #     N2 = int((n - 1) / 2.)
    #     endpoint = True

    # if shift:
    #     f = np.linspace(-N / 2., N / 2., n, endpoint=endpoint)
    # else:
    #     f = np.hstack((np.linspace(0, N / 2., N1, endpoint=endpoint),
    #                    np.linspace(-N / 2., 0, N2, endpoint=False)))
    # if norm:
    #     return th.tensor(f / n, dtype=dtype)
    # else:
    #     return th.tensor(f / (d * n), dtype=dtype)

    n = int(n)
    d = 1. / fs
    if norm:
        s = 1.0 / n
    else:
        s = 1.0 / (n * d)
    results = th.empty(n, dtype=int, device=device)
    N = (n - 1) // 2 + 1
    pp = th.arange(0, N, dtype=int, device=device)
    pn = th.arange(-(n // 2), 0, dtype=int, device=device)

    results[:N] = pp
    results[N:] = pn

    if shift:
        results = fftshift(results, axis=0)

    return results * s


def fftshift(x, axis=None):
    r"""Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int, optional
        Axes over which to shift. (Default is None, which shifts all axes.)

    Returns
    -------
    y : Tensor
        The shifted tensor.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    ::

        import numpy as np
        import torchsar as ts
        import torch as th

        x = [1, 2, 3, 4, 5, 6]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = ts.fftshift(x)
        print(y)

        x = [1, 2, 3, 4, 5, 6, 7]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = ts.fftshift(x)
        print(y)

        axis = (0, 1)  # axis = 0, axis = 1
        x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
        y = np.fft.fftshift(x, axis)
        print(y)
        x = th.tensor(x)
        y = ts.fftshift(x, axis)
        print(y)


        x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
        y = np.fft.fftshift(x, axis)
        print(y)
        x = th.tensor(x)
        y = ts.fftshift(x, axis)
        print(y)

    """

    if axis is None:
        axis = tuple(range(x.dim()))
    elif type(axis) is int:
        axis = tuple([axis])
    for a in axis:
        n = x.size(a)
        p = int(n / 2.)
        x = th.roll(x, p, dims=a)
    return x


def ifftshift(x, axis=None):
    r"""Shift the zero-frequency component back.

    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    axis : int, optional
        Axes over which to shift. (Default is None, which shifts all axes.)

    Returns
    -------
    y : Tensor
        The shifted tensor.

    See Also
    --------
    fftshift : The inverse of `ifftshift`.

    Examples
    --------
    ::

        import numpy as np
        import torchsar as ts
        import torch as th

        x = [1, 2, 3, 4, 5, 6]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = ts.fftshift(x)
        print(y)

        x = [1, 2, 3, 4, 5, 6, 7]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = ts.fftshift(x)
        print(y)

        axis = (0, 1)  # axis = 0, axis = 1
        x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
        y = np.fft.fftshift(x, axis)
        print(y)
        x = th.tensor(x)
        y = ts.fftshift(x, axis)
        print(y)


        x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
        y = np.fft.fftshift(x, axis)
        print(y)
        x = th.tensor(x)
        y = ts.fftshift(x, axis)
        print(y)

    """

    if axis is None:
        axis = tuple(range(x.dim()))
    elif type(axis) is int:
        axis = tuple([axis])
    for a in axis:
        n = x.size(a)
        p = int((n + 1) / 2.)
        x = th.roll(x, p, dims=a)
    return x


def padfft(X, nfft=None, axis=0, shift=False):
    r"""PADFT Pad array for doing FFT or IFFT

    PADFT Pad array for doing FFT or IFFT

    Parameters
    ----------
    X : Tensor
        Data to be padded.
    nfft : {number or None}
        Padding size.
    axis : int, optional
        Padding dimension. (the default is 0)
    shift : bool, optional
        Whether to shift the frequency (the default is False)

    Returns
    -------
    y : Tensor
        The padded tensor.
    """

    if axis is None:
        axis = 0

    Nx = X.size(axis)

    if nfft < Nx:
        raise ValueError('Output size is smaller than input size!')

    pad = list(X.size())

    Np = int(np.uint(nfft - Nx))

    if shift:
        pad[axis] = int(np.fix((Np + 1) / 2.))
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((Z, X), dim=axis)
        pad[axis] = Np - pad[axis]
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((X, Z), dim=axis)
    else:
        pad[axis] = Np
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((X, Z), dim=axis)

    return X


def fft(x, n=None, axis=0, norm="backward", shift=False):
    """FFT in torchsar

    FFT in torchsar.

    Parameters
    ----------
    x : {torch array}
        complex representation is supported. Since torch1.7 and above support complex array,
        when :attr:`x` is in real-representation formation(last dimension is 2, real, imag),
        we will change the representation in complex formation, after FFT, it will be change back.
    n : int, optional
        number of fft points (the default is None --> equals to signal dimension)
    axis : int, optional
        axis of fft (the default is 0, which the first dimension)
    norm : {None or str}, optional
        Normalization mode. For the forward transform (fft()), these correspond to:
        - "forward" - normalize by ``1/n``
        - "backward" - no normalization (default)
        - "ortho" - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)
    shift : bool, optional
        shift the zero frequency to center (the default is False)

    Returns
    -------
    y : {torch array}
        fft results torch array with the same type as :attr:`x`

    Raises
    ------
    ValueError
        nfft is small than signal dimension.
    """

    if norm is None:
        norm = 'backward'

    if (x.size(-1) == 2) and (not th.is_complex(x)):
        realflag = True
        x = th.view_as_complex(x)
        if axis < 0:
            axis += 1
    else:
        realflag = False

    d = x.size(axis)
    if n is None:
        n = d
    if d < n:
        x = padfft(x, n, axis, shift)
    elif d > n:
        raise ValueError('nfft is small than signal dimension!')

    if shift:
        y = thfft.fftshift(thfft.fft(thfft.fftshift(x, dim=axis),
                                     n=n, dim=axis, norm=norm), dim=axis)
    else:
        y = thfft.fft(x, n=n, dim=axis, norm=norm)

    if realflag:
        y = th.view_as_real(y)

    return y


def ifft(x, n=None, axis=0, norm="backward", shift=False):
    """IFFT in torchsar

    IFFT in torchsar, since ifft in torch only supports complex-complex transformation,
    for real ifft, we insert imaginary part with zeros (torch.stack((x,torch.zeros_like(x), dim=-1))),
    also you can use torch's rifft.

    Parameters
    ----------
    x : {torch array}
        both complex and real representation are supported. Since torch does not
        support complex array, when :attr:`x` is complex, we will change the representation
        in real formation(last dimension is 2, real, imag), after IFFT, it will be change back.
    n : int, optional
        number of ifft points (the default is None --> equals to signal dimension)
    axis : int, optional
        axis of ifft (the default is 0, which the first dimension)
    norm : bool, optional
        Normalization mode. For the backward transform (ifft()), these correspond to:
        - "forward" - no normalization
        - "backward" - normalize by ``1/n`` (default)
        - "ortho" - normalize by 1``/sqrt(n)`` (making the IFFT orthonormal)
    shift : bool, optional
        shift the zero frequency to center (the default is False)

    Returns
    -------
    y : {torch array}
        ifft results torch array with the same type as :attr:`x`

    Raises
    ------
    ValueError
        nfft is small than signal dimension.
    """

    if norm is None:
        norm = 'backward'

    if (x.size(-1) == 2) and (not th.is_complex(x)):
        realflag = True
        x = th.view_as_complex(x)
        if axis < 0:
            axis += 1
    else:
        realflag = False

    if shift:
        y = thfft.ifftshift(thfft.ifft(thfft.ifftshift(x, dim=axis),
                                       n=n, dim=axis, norm=norm), dim=axis)
    else:
        y = thfft.ifft(x, n=n, dim=axis, norm=norm)

    if realflag:
        y = th.view_as_real(y)

    return y


if __name__ == '__main__':

    print(th.__version__)
    nfft = 4
    ftshift = False
    x1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    print(x1.shape)
    y1 = np.fft.fft(x1, n=nfft, axis=1, norm=None)
    print(y1, y1.shape)
    x1 = np.fft.ifft(y1, n=nfft, axis=1, norm=None)
    print(x1)

    x2 = th.tensor(x1, dtype=th.float32)
    x2 = th.stack([x2, th.zeros(x2.size())], dim=-1)

    y2 = fft(x2, n=nfft, axis=1, norm=None, shift=ftshift)
    print(y2, y2.shape)
    x2 = ifft(y2, n=nfft, axis=1, norm=None, shift=ftshift)
    print(x2)
