#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 by Gongfan Fang, Zhejiang University. All rights reserved.
# @Note    : https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py


import torch
from torchsar.evaluation.ssims import _fspecial_gauss_1d, ssim, msssim


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MSSSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MSSSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return msssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )


if __name__ == '__main__':
    import torchsar as ts
    import torch
    from torch import optim
    from torch.autograd import Variable

    L = 255.0
    # L = 1.0
    ssim_value = 1e-3

    img1 = torch.zeros(1, 3, 512, 512)
    img1[:, :, 10:500, 10:500] = L / 2.0
    img2 = torch.rand(img1.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    X = Variable(img2, requires_grad=True)
    Y = Variable(img1, requires_grad=False)

    # reuse the gaussian kernel with SSIM & MS_SSIM.
    ssim_module = SSIM(data_range=255, size_average=True, channel=3)
    ms_ssim_module = MSSSIM(data_range=255, size_average=True, channel=3)



    optimizer = optim.Adam([img2], lr=0.01)

    while ssim_value < 0.999:
        optimizer.zero_grad()
        ssim_loss = 1 - ssim_module(X, Y)
        ms_ssim_loss = 1 - ms_ssim_module(X, Y)
        ssim_value = ssim_loss.item()
        print(ssim_value)
        ssim_loss.backward()
        ms_ssim_loss.backward()
        optimizer.step()
