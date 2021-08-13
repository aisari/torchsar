#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: spopoff
"""

import torch as th
from torch.nn.functional import relu, leaky_relu, max_pool2d, max_pool1d, dropout, dropout2d, upsample


def complex_relu(input, inplace=False):
    return th.stack((relu(input[..., 0], inplace),
                     relu(input[..., 1], inplace)), dim=-1)


def complex_leaky_relu(input, negative_slope=(0.01, 0.01), inplace=False):
    return th.stack((leaky_relu(input[..., 0], negative_slope[0], inplace),
                     leaky_relu(input[..., 1], negative_slope[1], inplace)), dim=-1)


def complex_max_pool2d(input, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):

    return th.stack((max_pool2d(input[..., 0], kernel_size, stride, padding, dilation, ceil_mode, return_indices),
                     max_pool2d(input[..., 1], kernel_size, stride, padding, dilation, ceil_mode, return_indices)), dim=-1)


def complex_max_pool1d(input, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):

    return th.stack((max_pool1d(input[..., 0], kernel_size, stride, padding, dilation, ceil_mode, return_indices),
                     max_pool1d(input[..., 1], kernel_size, stride, padding, dilation, ceil_mode, return_indices)), dim=-1)


def complex_dropout(input, p=0.5, training=True, inplace=False):
    return th.stack((dropout(input[..., 0], p, training, inplace),
                     dropout(input[..., 1], p, training, inplace)), dim=-1)


def complex_dropout2d(input, p=0.5, training=True, inplace=False):
    return th.stack((dropout2d(input[..., 0], p, training, inplace),
                     dropout2d(input[..., 1], p, training, inplace)), dim=-1)


def complex_upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return th.stack((upsample(input[..., 0], size, scale_factor, mode, align_corners),
                     upsample(input[..., 1], size, scale_factor, mode, align_corners)), dim=-1)

