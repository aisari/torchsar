#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


import numpy as np
import torch as th
import torchsar as ts


def sinc(x):
    flag = False
    if type(x) is not th.Tensor:
        flag = True
        x = th.tensor(x)
    x = th.where(x.abs() < 1e-32, th.tensor(1., dtype=x.dtype, device=x.device), th.sin(ts.PI * x) / (ts.PI * x))

    if flag:
        return x.item()
    else:
        return x


def nextpow2(x):
    if x == 0:
        y = 0
    else:
        y = int(np.ceil(np.log2(x)))

    return y


def prevpow2(x):
    if x == 0:
        y = 0
    else:
        y = int(np.floor(np.log2(x)))
    return y


def ebemulcc(A, B):
    """Element-by-element complex multiplication

    like A .* B in matlab

    Parameters
    ----------
    A : {torch array}
        any size torch array, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).
    B : {torch array}
        any size torch array, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).
        :attr:`B` has the same size as :attr:`A`.

    Returns
    -------
    torch array
        result of element-by-element complex multiplication with the same repesentation as :attr:`A`.
    """

    if th.is_complex(A) and th.is_complex(B):
        return A.real * B.real - A.imag * B.imag + 1j * (A.real * B.imag + A.imag * B.real)
    else:
        return th.stack((A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1], A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]), dim=-1)


def mmcc(A, B):
    """Complex matrix multiplication

    like A * B in matlab

    Parameters
    ----------
    A : {torch array}
        any size torch array, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).
    B : {torch array}
        any size torch array, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).

    Returns
    -------
    torch array
        result of complex multiplication with the same repesentation as :attr:`A`.
    """

    if th.is_complex(A) and th.is_complex(B):
        return th.mm(A.real, B.real) - th.mm(A.imag, B.imag) + 1j * (th.mm(A.real, B.imag) + th.mm(A.imag, B.real))
    else:
        return th.stack((th.mm(A[..., 0], B[..., 0]) - th.mm(A[..., 1], B[..., 1]), th.mm(A[..., 0], B[..., 1]) + th.mm(A[..., 1], B[..., 0])), dim=-1)


def matmulcc(A, B):
    """Complex matrix multiplication

    like A * B in matlab

    Parameters
    ----------
    A : {torch array}
        any size torch array, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).
    B : {torch array}
        any size torch array, both complex and real representation are supported.
        For real representation, the last dimension is 2 (the first --> real part, the second --> imaginary part).

    Returns
    -------
    torch array
        result of complex multiplication with the same repesentation as :attr:`A`.
    """

    if th.is_complex(A) and th.is_complex(B):
        return th.matmul(A.real, B.real) - th.matmul(A.imag, B.imag) + 1j * (th.matmul(A.real, B.imag) + th.matmul(A.imag, B.real))
    else:
        return th.stack((th.matmul(A[..., 0], B[..., 0]) - th.matmul(A[..., 1], B[..., 1]), th.matmul(A[..., 0], B[..., 1]) + th.matmul(A[..., 1], B[..., 0])), dim=-1)


def conj(X):

    if th.is_complex(X):
        return th.conj(X)
    elif X.size(-1) == 2:
        return th.stack((X[..., 0], -X[..., 1]), dim=-1)
    else:
        raise TypeError('Not known type! Only real and imag representions are supported!')


def absc(X):

    if X.size(-1) == 2:
        X = X.pow(2).sum(axis=-1).sqrt()
    else:
        X = th.abs(X)

    return X


if __name__ == '__main__':

    Ar = th.randn((3, 3, 2))
    Br = th.randn((3, 3, 2))

    Ac = th.view_as_complex(Ar)
    Bc = th.view_as_complex(Br)

    Mr = ebemulcc(Ar, Br)
    Mc = th.view_as_real(Ac * Bc)
    print(th.sum(Mr - Mc))

    Mc = mmcc(Ac, Bc)
    Mr = mmcc(Ar, Br)
    Mc = th.view_as_real(Mc)
    print(th.sum(Mr - Mc))

    Mc = matmulcc(Ac, Bc)
    Mr = matmulcc(Ar, Br)
    Mc = th.view_as_real(Mc)
    print(th.sum(Mr - Mc))
