#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import math
import torch as th
import numpy as np
from torchsar.utils.const import *

AirboneSatelliteHeightBoundary = 60e3  # 60 km


def compute_sar_parameters(pdict, islog=False):

    trmod = 'WithTp'
    trmod = 'WithoutTp'

    if 'GeometryMode' not in pdict:
        pdict['GeometryMode'] = 'SceneGeometry'
    if pdict['GeometryMode'] is None:
        pdict['GeometryMode'] = 'BeamGeometry'

    if islog:
        print("---In compute_sar_parameters------------------------------------------------------")

    Re = Ree
    Rs = pdict['H'] + Re
    pdict['Rs'] = Rs
    if 'Vr' not in pdict or pdict['Vr'] is None:
        pdict['Vr'] = pdict['V']
    if 'V' not in pdict or pdict['V'] is None:
        pdict['V'] = pdict['Vr']

    if pdict['H'] < AirboneSatelliteHeightBoundary:
        print("---Airbone.")
        pdict['Vs'] = pdict['V']
        pdict['Vr'] = pdict['V']
        pdict['Vg'] = pdict['V']
        pdict['To'] = -1
        pdict['Ws'] = -1
    else:
        # print("---Satellite.")
        # Earth Bending Geometry (Vg < Vr < Vs, Vr = \sqrt{Vg*Vs})
        pdict['To'] = 2 * PI * math.sqrt((Rs)**3 / Ge)
        pdict['Ws'] = 2 * PI / pdict['To']
        pdict['Vs'] = Rs * pdict['Ws']
        pdict['Vr'] = pdict['V']
        # Equivalent velocity (Linear geometry)
        pdict['Vg'] = pdict['Vr']**2 / pdict['Vs']  # ground velocity
    Vr = pdict['Vs'] * math.sqrt(Re / Rs)
    if islog:
        print("---Computed equivalent velocity is %.4f, setted is %.4f (utilized)" %
              (Vr, pdict['Vr']))
    pdict['Vs'] = Vr / math.sqrt(Re / Rs)  # reset

    if 'PlatCenter' not in pdict or pdict['PlatCenter'] is None:
        pdict['PlatCenter'] = [0, 0, pdict['H'], 0, pdict['Vr'], 0]
    if 'SceneCenter' not in pdict or pdict['SceneCenter'] is None:
        pdict['SceneCenter'] = [0, 0, 0]

    # As: squint angle in earth curve geometry, Ar: squint angle in line geometry
    pdict['Ar'] = (pdict['Vs'] / pdict['Vr']) * pdict['As']
    if 'PRF' not in pdict or pdict['PRF'] is None:
        pdict['PRF'] = pdict['Fsa']
    if 'Fsa' not in pdict or pdict['Fsa'] is None:
        pdict['Fsa'] = pdict['PRF']
    if 'Fsr' not in pdict or pdict['Fsr'] is None:
        pdict['Fsr'] = pdict['Fs']
    if 'Fs' not in pdict or pdict['Fs'] is None:
        pdict['Fs'] = pdict['Fsr']

    SceneCenter = pdict['SceneCenter']  # [x, y, z, vx, vy, vz]
    pdict['Wl'] = C / pdict['Fc']
    pdict['Br'] = abs(pdict['Kr'] * pdict['Tp'])
    Na, Nr = pdict['EchoSize'][0:2]

    pdict['Ta'], pdict['Tr'] = (Na * 1.0) / pdict['Fsa'], (Nr * 1.0) / pdict['Fsr']
    pdict['Tsa'], pdict['Tsr'] = 1. / pdict['Fsa'], 1.0 / pdict['Fsr']

    if 'Aon' not in pdict or pdict['Aon'] is None:
        pdict['Aon'] = math.acos(pdict['H'] / pdict['Rnear'])

    pdict['Rnear'] = pdict['H'] / (math.cos(pdict['Aon']) + EPS)
    if trmod == 'WithoutTp':
        pdict['Rfar'] = pdict['Rnear'] + pdict['Tr'] * C / 2
    if trmod == 'WithTp':
        pdict['Rfar'] = pdict['Rnear'] + (pdict['Tr'] - pdict['Tp']) * C / 2
        # pdict['Rfar'] = pdict['Rnear'] + pdict['Tr'] * C / 2

    pdict['tnear'] = pdict['Rnear'] / C
    pdict['tfar'] = pdict['Rfar'] / C
    pdict['SWST'] = 2. * pdict['tnear']  # Sampling Window Start Time, SWST
    pdict['Abr'] = math.acos(pdict['H'] / pdict['Rfar']) - abs(pdict['Aon'])
    pdict['Ad'] = np.sign(pdict['Aon']) * (PI / 2. - (abs(pdict['Aon']) + pdict['Abr'] / 2.))
    pdict['FPr'] = math.sqrt(pdict['Rfar']**2 + pdict['Rnear']**2 - 2. *
                             pdict['Rnear'] * pdict['Rfar'] * math.cos(pdict['Abr']) + EPS)
    # pdict['Rsc'] = 0.5 * math.sqrt(2. * pdict['Rfar']**2 + 2. * pdict['Rnear']**2 - pdict['FPr']**2 + EPS)
    pdict['Rsc'] = math.sqrt(
        pdict['H']**2 + ((pdict['Rfar']**2 - pdict['Rnear']**2) / 2 / pdict['FPr'])**2)

    pdict['Rs0'] = abs(pdict['Rsc'] * math.cos(pdict['Ar']))

    pdict['Ysc'] = pdict['Rsc'] * math.sin(pdict['Ar'])
    pdict['Xsc'] = math.sqrt(abs(pdict['Rs0']**2 - pdict['H']**2))
    # print("[[[[Rs0, Rsc, H, Xsc", pdict['Rs0'], pdict['Rsc'], pdict['H'], pdict['Xsc'])

    pdict['Ynear'] = pdict['Rnear'] * math.sin(pdict['Ar'])
    pdict['Yfar'] = pdict['Rfar'] * math.sin(pdict['Ar'])
    pdict['Xnear'] = math.sqrt(abs((pdict['Rnear']**2 - pdict['H']**2) - pdict['Ynear']**2 + EPS))
    pdict['Xfar'] = math.sqrt(abs((pdict['Rfar']**2 - pdict['H']**2) - pdict['Yfar']**2 + EPS))
    SceneArea = [0, 0, 0, 0]
    SceneArea[0] = pdict['Xnear'] - pdict['Xsc']
    SceneArea[1] = pdict['Xfar'] - pdict['Xsc']
    SceneArea[2] = -pdict['Ta'] * pdict['Vr'] / 2.0
    SceneArea[3] = pdict['Ta'] * pdict['Vr'] / 2.0
    # print("Rnear, Rfar, Xnear, Xfar", pdict['Rnear'], pdict['Rfar'], pdict['Xnear'], pdict['Xfar'], "]]]]]]")
    SceneCenter[0], SceneCenter[1] = pdict['Xsc'], pdict['Ysc']
    pdict['SceneCenter'] = SceneCenter
    pdict['SceneArea'] = SceneArea
    pdict['Na'], pdict['Nr'] = pdict['EchoSize'][0:2]

    pdict['FPr0'] = pdict['H'] / \
        math.tan(pdict['Ad'] + pdict['Abr'] / 2.) - pdict['H'] / math.tan(pdict['Ad'])
    pdict['FPr1'] = pdict['H'] / \
        math.tan(pdict['Ad'] - pdict['Abr'] / 2.) - pdict['H'] / math.tan(pdict['Ad'])
    BeamArea = (pdict['FPr0'], pdict['FPr1'], -pdict['Ta'] *
                pdict['Vg'] / 2.0, pdict['Ta'] * pdict['Vg'] / 2.0)
    pdict['Rbc'] = pdict['H'] / (math.sin(pdict['Ad']) + EPS)
    pdict['Rb0'] = pdict['Rbc'] * math.cos(pdict['Ar'])

    pdict['Ybc'] = pdict['Rbc'] * math.sin(pdict['Ar'])
    pdict['Xbc'] = math.sqrt(abs(pdict['Rb0']**2 - pdict['H']**2))
    pdict['BeamCenter'] = [pdict['Xbc'], pdict['Ybc'], 0.0]
    pdict['BeamArea'] = BeamArea

    if trmod == 'WithoutTp':
        pdict['tstart'] = 2 * pdict['tnear']
        pdict['tend'] = 2 * pdict['tfar']
    if trmod == 'WithTp':
        # tmid = (pdict['Rfar'] + pdict['Rnear']) / C
        pdict['tstart'] = 2 * pdict['tnear']
        pdict['tend'] = 2 * pdict['tfar'] + pdict['Tp']
        # pdict['tstart'] = 2 * pdict['tnear'] - pdict['Tp'] / 2.
        # pdict['tend'] = 2 * pdict['tfar'] + pdict['Tp'] / 2.
        # pdict['tstart'] = tmid - pdict['Tr'] / 2.
        # pdict['tend'] = tmid + pdict['Tr'] / 2.

    # assert (pdict['tend'] - pdict['tstart']) > pdict[
    #     'Tp'], "~~~Range sampling time %.4f(us) too small, should larger than pulse width %.4f(us)" % (pdict['Tr'] * 1e6, pdict['Tp'] * 1e6)

    pdict['ta'] = th.linspace(-pdict['Ta'] / 2., pdict['Ta'] / 2., Na).reshape(Na, 1)
    pdict['tr'] = th.linspace(pdict['tstart'], pdict['tend'], Nr).reshape(1, Nr)

    pdict['Lsar'] = (C / pdict['Fc']) * pdict['Rnear'] / pdict['La']
    pdict['Tsar'] = pdict['Lsar'] / pdict['Vr']
    pdict['Nsar'] = int(pdict['Tsar'] * pdict['Fsa'])

    pdict['Ka'] = (-2.0 * pdict['Vr']**2 * math.cos(pdict['Ar'])**3) / \
        (pdict['Wl'] * pdict['Rnear'])
    pdict['Ba'] = abs(pdict['Ka'] * pdict['Tsar'])

    if islog:
        if pdict['Fsa'] / pdict['Ba'] < 1.1 or pdict['Fsa'] / pdict['Ba'] > 2.0:
            print(
                "---Azimuth sampling rate (Fsa=%f) should be in range [1.1, 2.0]*Ba=[1.1, 2.0]*%f!" % (pdict['Fsa'], pdict['Ba']))
        if pdict['Fsr'] / pdict['Br'] < 1.1 or pdict['Fsr'] / pdict['Br'] > 2.0:
            print(
                "---Range sampling rate (Fsr=%.4e) should be in range [1.1, 2.0]*Br=[1.1, 2.0]*%.4e!" % (pdict['Fsr'], pdict['Br']))

    Fsa = 1. / (pdict['Tr'] + pdict['Tp'])
    assert (pdict['Tsa'] - pdict['Tp']) > pdict['Tr'], "Azimuth sampling rate (Fsa=%.4f) should be lower than %.4f to avoid range ambiguity (1.0 ÷ Fsa - Tp > Tr)" % (pdict['Fsa'], Fsa)

    pdict['osfa'] = pdict['Fsa'] / pdict['Ba']
    pdict['osfr'] = pdict['Fsr'] / pdict['Br']
    print(pdict['Fsa'], pdict['Ba'])
    print(pdict['Fsr'], pdict['Br'])
    assert pdict['osfa'] > pdict['osfr'], "Over sampling factor in azimuth=%.4f should be larger than range=%.4f" % (pdict['osfa'], pdict['osfr'])

    if pdict['GeometryMode'] == 'SceneGeometry':
        pdict['R0'] = pdict['Rs0']

    if pdict['GeometryMode'] == 'BeamGeometry':
        pdict['R0'] = pdict['Rb0']
    pdict['tac'] = -pdict['R0'] * math.tan(pdict['Ar']) / pdict['Vr']

    if 'fdc' not in pdict:
        pdict['fdc'] = 2.0 * pdict['Vr'] * math.sin(pdict['Ar']) / pdict['Wl']

    pdict['Bdop'] = 0.886 * (2 * pdict['Vs'] * math.cos(pdict['As']) / pdict['La'])

    if pdict['H'] < AirboneSatelliteHeightBoundary:
        pdict['Ai'] = PI / 2. - pdict['Ad']
    else:
        pdict['Ao'] = math.acos((Rs**2 + Rea**2 - pdict['Rbc']**2) / (2. * Rs * Rea))
        pdict['Ai'] = (PI / 2. - pdict['Ad']) + pdict['Ao']

    pdict['AzimuthResolution'] = pdict['La'] / 2.
    pdict['SlantRangeResolution'] = C / 2. / pdict['Br']
    pdict['GroundRangeResolution'] = pdict['SlantRangeResolution'] / (math.sin(pdict['Ai']) + EPS)
    pdict['ImagePixelResolution'] = [(pdict['SceneArea'][3] - pdict['SceneArea'][2]) /
                                     pdict['Na'], (pdict['SceneArea'][1] - pdict['SceneArea'][0]) / pdict['Nr']]

    if islog:
        print("---SAR platform height: %.4f(m)" % pdict['H'])
        print("---SAR platform velocity: %.4f(m/s)" % pdict['V'])
        print("---Carrier frequency: %.4f(Hz)" % pdict['Fc'])
        print("---Wavelength: %.4f(m)" % pdict['Wl'])
        print("---Squint angle: %.4f(degree)" % (pdict['As'] * 180 / PI))
        print("---Depression angle: %.4f(degree)" % (pdict['Ad'] * 180 / PI))
        print("---Antenna range beamwidth: %.4f(degree)" % (pdict['Abr'] * 180 / PI))
        print("---off-nadir angle: %.4f(degree)" % (pdict['Aon'] * 180 / PI))
        print("---Azimuth antenna length: %.4f(m)" % pdict['La'])
        print("---Azimuth bandwidth: %.4f(Hz)" % pdict['Ba'])
        print("---Azimuth sampling rate: %.4f(Hz)" % pdict['Fsa'])
        print("---Azimuth over sampling factor: %.4f" % pdict['osfa'])
        print("---Azimuth sampling time: %.4f(s)" % pdict['Ta'])
        print("---Azimuth doppler bandwidth: %.4f(Hz)" % pdict['Bdop'])
        print("---Chirp rate: %.4f(Hz)" % pdict['Kr'])
        print("---Transmitted pulse width: %.4f(us)" % (pdict['Tp'] * 1e6))
        print("---Range bandwidth: %.4f(MHz)" % (pdict['Br'] * 1e-6))
        print("---Range sampling rate: %.4f(MHz)" % (pdict['Fsr'] * 1e-6))
        print("---Range sampling time: %.4f(us)" % (pdict['Tr'] * 1e6))
        print("---Range over sampling factor: %.4f" % pdict['osfr'])
        print("---Range start sampling time: %.4f(us)" % (pdict['tstart'] * 1e6))
        print("---Range stop sampling time: %.4f(us)" % (pdict['tend'] * 1e6))
        print("---Scene area (m): [%.4f, %.4f, %.4f, %.4f]" % (pdict['SceneArea'][
              0], pdict['SceneArea'][1], pdict['SceneArea'][2], pdict['SceneArea'][3]))
        print("---Scene center (m): [%.4f, %.4f]" %
              (pdict['SceneCenter'][0], pdict['SceneCenter'][1]))
        print("---Beam area (m): [%.4f, %.4f, %.4f, %.4f]" % (pdict['BeamArea'][
            0], pdict['BeamArea'][1], pdict['BeamArea'][2], pdict['BeamArea'][3]))
        print("---Beam center (m): [%.4f, %.4f]" %
              (pdict['BeamCenter'][0], pdict['BeamCenter'][1]))
        print("---Echo size: Azimuth×Range=%d × %d" % (pdict['Na'], pdict['Nr']))
        print("---Azimuth resolution: %.4f(m)" % (pdict['AzimuthResolution']))
        print("---Slant range resolution: %.4f(m)" % (pdict['SlantRangeResolution']))
        print("---Ground range resolution: %.4f(m)" % (pdict['GroundRangeResolution']))
        print("---Image pixel resolution: %.4f(m) × %.4f(m)" %
              (pdict['ImagePixelResolution'][0], pdict['ImagePixelResolution'][1]))
        print("---Minimum slant range: %.4f(m)" % (pdict['Rnear']))
        print("---Maximum slant range: %.4f(m)" % (pdict['Rfar']))
        print("---Sampling window start time: %.4f(us)" % (pdict['SWST'] * 1e6))
        print("---Synthetic aperture size: %.4f(m)" % (pdict['Lsar']))
        print("---Synthetic aperture time: %.4f(s)" % (pdict['Tsar']))
        print("---Synthetic aperture pixels: %d" % (pdict['Nsar']))
        print("---Out compute_sar_parameters------------------------------------------------------")

    return pdict
