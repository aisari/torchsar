#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-06 10:28:33
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

from torchsar.utils.const import *

SENSOR = {

    'Air1': {
        'Fc': 5.3e9,  # Hz
        'H': 10000,  # height in m
        'V': 150.0,  # velocity in m/s
        'Tp': 2.5e-6,  # Range pulse length in seconds
        'Kr': 20e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 12,  # antenna length (range) in m
        'La': 3.0,  # antenna length (azimuth) in m
        'PRF': 120.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 60.0e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [256, 320],
        'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        # 'As': 2 * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        # 'Aon': 59.3392 * PI / 180.0,  # off-nadir angle |\
        'Aon': 59.33 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Air1',
    },

    'Air2': {
        'Fc': 9.4e9,  # Hz
        'H': 8000,  # height in m
        'V': 200.0,  # velocity in m/s
        'Tp': 1e-6,  # Range pulse length in seconds
        'Kr': 160e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 12,  # antenna length (range) in m
        'La': 3.0,  # antenna length (azimuth) in m
        'PRF': 90.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 96e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [128, 128],
        'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        # 'As': 0.5 * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 58.7 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Air2',
    },

    'Air3': {
        'Fc': 9.4e9,  # Hz
        'H': 8000,  # height in m
        'V': 200.0,  # velocity in m/s
        'Tp': 1e-6,  # Range pulse length in seconds
        'Kr': 160e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 12,  # antenna length (range) in m
        'La': 3,  # antenna length (azimuth) in m
        'PRF': 300.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 192.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [256, 256],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 67.7 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Air3',
    },

    'Air4': {
        'Fc': 9.4e9,  # Hz
        'H': 8000,  # height in m
        'V': 200.0,  # velocity in m/s
        'Tp': 1e-6,  # Range pulse length in seconds
        'Kr': 80e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 12,  # antenna length (range) in m
        'La': 6,  # antenna length (azimuth) in m
        'PRF': 100.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 96.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [64, 64],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 67.7 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Air4',
    },

    'Air5': {
        'Fc': 9.4e9,  # Hz
        'H': 8000,  # height in m
        'V': 200.0,  # velocity in m/s
        'Tp': 1e-6,  # Range pulse length in seconds
        'Kr': 80e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 12,  # antenna length (range) in m
        'La': 6,  # antenna length (azimuth) in m
        'PRF': 100.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 96.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [128, 128],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 67.7 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Air5',
    },

    'Space3': {
        'Fc': 1.27e9,  # Hz
        'H': 800000,  # height in m
        'V': 7100.0,  # velocity in m/s
        'Tp': 5e-6,  # Range pulse length in seconds
        'Kr': 8e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 12,  # antenna length (range) in m
        'La': 15.0,  # antenna length (azimuth) in m
        'PRF': 1800.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 48.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [2048, 2048],
        'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        # 'As': 0.05 * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 23.4 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Space3',
    },

    'ALOS': {
        'Fc': 1.27e9,  # Hz
        'H': 800000,  # height in m
        'V': 7153.0,  # velocity in m/s
        'Tp': 27e-6,  # Range pulse length in seconds
        'Kr': -28.e6 / 27.0e-6,  # FM rate of radar pulse (Hz/s)
        'Lr': 2.9,  # antenna length (range) in m
        'La': 8.9,  # antenna length (azimuth) in m
        'PRF': 1912.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 32.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [512, 1024],
        # 'EchoSize': [35345, 12040],
        'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        # 'As': 0.05 * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 34.3 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Space3',
    },

    'Sim1': {
        'Fc': 5.3e9,  # Hz
        'H': 1000,  # height in m
        'V': 150.0,  # velocity in m/s
        'Tp': 2.5e-6,  # Range pulse length in seconds
        'Kr': 40.e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 2,  # antenna length (range) in m
        'La': 3,  # antenna length (azimuth) in m
        'PRF': 180.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 120.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [512, 512],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 67.7 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Sim1',
    },

    'Sim2': {
        'Fc': 5.3e9,  # Hz
        'H': 1000,  # height in m
        'V': 150.0,  # velocity in m/s
        'Tp': 2.5e-6,  # Range pulse length in seconds
        'Kr': 40.e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 2,  # antenna length (range) in m
        'La': 3,  # antenna length (azimuth) in m
        'PRF': 180.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 120.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [256, 256],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 67.7 * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Sim2',
    },

    'Sim3': {
        'Fc': 5.3e9,  # Hz
        'H': 10000,  # height in m
        'V': 150.0,  # velocity in m/s
        'Tp': 0.5e-6,  # Range pulse length in seconds
        'Kr': 160e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 2,  # antenna length (range) in m
        'La': 3,  # antenna length (azimuth) in m
        'PRF': 160.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 96.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [128, 128],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 45. * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Sim3',
    },

    'Sim4': {
        'Fc': 5.3e9,  # Hz
        'H': 1000,  # height in m
        'V': 150.0,  # velocity in m/s
        'Tp': 0.5e-6,  # Range pulse length in seconds
        'Kr': 160.e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 2,  # antenna length (range) in m
        'La': 3,  # antenna length (azimuth) in m
        'PRF': 80.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 48.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [64, 64],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 45. * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Sim4',
    },

    'Sim5': {
        'Fc': 5.3e9,  # Hz
        'H': 1100,  # height in m
        'V': 150.0,  # velocity in m/s
        'Tp': 0.5e-6,  # Range pulse length in seconds
        'Kr': 80.e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 2,  # antenna length (range) in m
        'La': 5,  # antenna length (azimuth) in m
        'PRF': 100.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 48.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [64, 64],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 30. * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Sim5',
    },

    'Sim6': {
        'Fc': 5.3e9,  # Hz
        'H': 1100,  # height in m
        'V': 150.0,  # velocity in m/s
        'Tp': 0.5e-6,  # Range pulse length in seconds
        'Kr': 80.e+12,  # FM rate of radar pulse (Hz/s)
        'Lr': 2,  # antenna length (range) in m
        'La': 5,  # antenna length (azimuth) in m
        'PRF': 100.0,  # Hz
        # ADC sampling frequency, can be None: Ts = 1 / (1.2*self._sensor['B'])
        'Fs': 48.e6,

        'StateVector': None,
        'PlatCenter': None,  # [x, y, z], None: default --> [0, 0, H]
        'SceneCenter': None,
        # SceneArea:[xmin,xmax,ymin,ymax], unit: m
        'SceneArea': None,
        'EchoSize': [32, 32],
        # 'As': 0.0 * PI / 180.0,  # squint angle in earth curve geometry
        'As': 0. * PI / 180.0,  # squint angle in earth curve geometry
        'Ad': None,  # depression angle
        'Aon': 30. * PI / 180.0,  # off-nadir angle |\
        'Aba': None,  # antenna azimuth beamwidth
        'Abr': None,  # antenna range beamwidth
        'Name': 'Sim6',
    },

}
