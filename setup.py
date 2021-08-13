#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2030, Zhi Liu.  All rights reserved.

from os import path as os_path
import torchsar
from setuptools import setup
from setuptools import find_packages
from distutils.core import setup as cysetup
# from Cython.Build import cythonize


this_dir = os_path.abspath(os_path.dirname(__file__))

def read_file(filename):
    with open(os_path.join(this_dir, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]

long_description = read_file('README.md'),
long_description_content_type = "text/markdown",

setup(name='torchsar',
      version=torchsar.__version__,
      description="PyTorch Tools for Synthetic Aperture Radar.",
      author='Zhi Liu',
      author_email='zhiliu.mind@gmail.com',
      url='https://sar.iridescent.ink/',
      download_url='https://github.com/aisari/torchsar',
      license='MIT',
      packages=find_packages(),
      install_requires=read_requirements('requirements.txt'),
      include_package_data=True,
      keywords=['PyTorch', 'Synthetic Aperture Radar', 'Machine Learning', 'Deep Learning'],
      # ext_modules=cythonize([this_dir + '/torchsar/misc/draw_shapes.py', this_dir + '/torchsar/misc/mapping_operation.py', this_dir + '/torchsar/misc/sampling.py'])
)


