#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import division, print_function, absolute_import
import h5py
import json
import yaml
import numpy as np
import scipy.io as scio


def loadyaml(filepath, field=None):
    """Load a yaml file.

    Parameters
    ----------
    filepath : str
        The file path string.
    field : None, optional
        The string of field that want to be loaded.

    """
    f = open(filepath, 'r', encoding='utf-8')
    if field is None:
        if int(yaml.__version__[0]) > 3:
            data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            data = yaml.load(f)
    else:
        if int(yaml.__version__[0]) > 3:
            data = yaml.load(f, Loader=yaml.FullLoader)[field]
        else:
            data = yaml.load(f)
    return data


def loadjson(filepath, field=None):
    """Summary

    Parameters
    ----------
    filepath : str
        The file path string.
    field : None, optional
        The string of field that want to be loaded.

    """
    with open(filepath, 'r', encoding='utf-8') as f:
        if field is None:
            data = json.load(f)
        else:
            data = json.load(f)[field]
    return data


def loadmat(filepath):

    return scio.loadmat(filepath)


def savemat(filepath, mdict, fmt='5', dtype=None):
    for k, v in mdict.items():
        if np.iscomplex(v).any() and np.ndim(v) > 1:
            mdict[k] = np.array(
                [np.real(v), np.imag(v)]).transpose(1, 2, 0)
            mdict[k] = mdict[k].astype('float32')
    scio.savemat(filepath, mdict, format=fmt)

    return 0


def _create_group_dataset(group, mdict):
    for k, v in mdict.items():
        if k in group.keys():
            del group[k]

        if type(v) is dict:
            subgroup = group.create_group(k)
            _create_group_dataset(subgroup, v)
        else:
            if v is None:
                v = []
            group.create_dataset(k, data=v)


def _read_group_dataset(group, mdict, keys=None):
    if keys is None:
        for k in group.keys():
            if type(group[k]) is h5py.Group:
                exec(k + '={}')
                _read_group_dataset(group[k], eval(k))
                mdict[k] = eval(k)
            else:
                mdict[k] = group[k][()]
    else:
        for k in keys:
            if type(group[k]) is h5py.Group:
                exec(k + '={}')
                _read_group_dataset(group[k], eval(k))
                mdict[k] = eval(k)
            else:
                mdict[k] = group[k][()]


def loadh5(filepath, keys=None):
    """load h5 file

    load all the data from a ``.h5`` file.

    Parameters
    ----------
    filepath : str
        File's full path string.

    Returns
    -------
    D : dict
        The loaded data in ``dict`` type.

    """

    f = h5py.File(filepath, 'r')
    D = {}

    _read_group_dataset(f, D, keys)

    f.close()
    return D


def saveh5(filepath, mdict, mode='w'):
    """save data to h5 file

    save data to ``.h5`` file

    Parameters
    ----------
    filepath : str
        filepath string
    mdict : dict
        each dict is store in group, the elements in dict are store in dataset
    mode : str
        save mode, ``'w'`` for write, ``'a'`` for add.

    Returns
    -------
    number
        0 --> all is well.
    """

    f = h5py.File(filepath, mode)

    _create_group_dataset(f, mdict)

    f.close()
    return 0


def mvkeyh5(filepath, ksf, kst, sep='.'):
    """rename keys in ``.h5`` file

    Parameters
    ----------
    filepath : str
        The file path string
    ksf : list
        keys from list, e.g. ['a.x', 'b.y']
    kst : list
        keys to list, e.g. ['a.1', 'b.2']
    sep : str, optional
        The separate pattern, default is ``'.'``

    Returns
    -------
    0
        All is ok!
    """
    ksf = [ksf] if type(ksf) is not list else ksf
    kst = [kst] if type(kst) is not list else kst
    f = h5py.File(filepath, 'a')
    for keyf, keyt in zip(ksf, kst):
        keyf = keyf.split(sep)
        keyt = keyt.split(sep)
        grp = f
        for kf, kt in zip(keyf[:-1], keyt[:-1]):
            grp = grp[kf]
        grp.create_dataset(keyt[-1], data=grp[keyf[-1]][()])
        del grp[keyf[-1]]
    f.close()
    return 0


if __name__ == '__main__':

    a = np.random.randn(3, 4)
    b = 10
    c = [1, 2, 3]
    d = {'1': 1, '2': a}
    s = 'Hello, the future!'
    t = (0, 1)

    saveh5('./data.h5', {'a': {'x': a}, 'b': b, 'c': c, 'd': d, 's': s})
    data = loadh5('./data.h5', keys=['a', 's'])
    print(data.keys())

    print("==========")
    # saveh5('./data.h5', {'t': t}, 'w')
    saveh5('./data.h5', {'t': t}, 'a')
    saveh5('./data.h5', {'t': (2, 3, 4)}, 'a')
    data = loadh5('./data.h5')

    for k, v in data.items():
        print(k, v)

    mvkeyh5('./data.h5', ['a.x'], ['a.1'])
    data = loadh5('./data.h5')

    for k, v in data.items():
        print(k, v)
