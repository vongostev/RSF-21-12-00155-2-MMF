# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:55:34 2021

@author: vonGostev
"""
import __init__
import numpy as np
import matplotlib.pyplot as plt
from lightprop2d import um
from scipy.interpolate import UnivariateSpline
from scipy.linalg import lstsq


def _n(x):
    return x / np.max(x)


def _c(y, X, r=-1):
    if r > 0:
        y[np.abs(X) > r] = 0
    return y


def FWHM(n, data1d):
    npeak = np.argmax(data1d)
    spline = UnivariateSpline(np.arange(n), data1d - np.max(data1d) / 2, s=0)
    r = spline.roots()
    if len(r) > 1:
        return abs(r[1] - r[0])
    elif len(r) == 1:
        return 2 * abs(r[0] - npeak)
    return 0


def lstsq_approx(X, Y):
    A = np.vstack((X, np.ones_like(X))).T
    kx, bx = lstsq(A, Y)[0]
    return kx, bx


SAVE = False
xlabel = 'Поперечная координата, мкм'
npoints = 2 ** 8
np_half = 2 ** 7
index_type = 'GRIN'
mod_type = 'slm'
fiber_type = 'mmf'
data_dir = 'mmf'
curvature = 10
_loaded_data = np.load(
    f'{data_dir}/cohdata_dist_011022_{fiber_type}_{mod_type}_{curvature}.npz',
    allow_pickle=True)
loaded_data = _loaded_data[index_type].tolist()

area_size = loaded_data['params']['area_size']
bounds = [-area_size / 2, area_size / 2]
core_radius = loaded_data['params']['core_radius']
wl = 0.632

distances = np.array([0, 20, 40, 60, 80, 100, 150,
                     200, 400, 1000, 3000, 10000]) * um
expands = [1] * 6 + [2] * 1 + [1] * 2 + [2] * 3

cfs = [v for k, v in loaded_data.items() if k.startswith('o__cf')]

widths = []
for i, cf in enumerate(cfs):
    _L = area_size * np.prod(expands[:i+1])
    width1 = np.sum(cf[np_half] >= 1 / 2) / npoints * _L
    width2 = FWHM(npoints, cf[np_half]) / npoints * _L
    widths.append(max(width1, width2))

exp_distances = np.array(
    [20, 40, 60, 80, 100, 150, 200, 1000, 3000, 10000]) * um
exp_rx = np.array([11.2, 11.4, 12.0, 12.7, 13.8, 16.3,
                  20.5, 36.78, 84.15, 302.175]) * 5.2 / 25.5
th_distances = np.sort(np.concatenate((distances, exp_distances)))
th_distances = np.linspace(th_distances[0], th_distances[-1], 100)
k_exp, b_exp = lstsq_approx(exp_distances, exp_rx)
plt.plot(th_distances, widths[0] + wl * th_distances / um / 31.25 * 0.16, '--',
         color='black', label='Теор. ', zorder=0)
plt.scatter(distances, widths, color='r', label='Мод.', marker='1', s=100)
plt.scatter(exp_distances, exp_rx, label='Эксп.', marker='2', zorder=-1, s=100)
plt.plot(th_distances, b_exp + k_exp * th_distances, '--', color='gray',
         label='Эксп. МНК')
plt.xscale('log')
plt.xlabel('Расстояние, см.')
plt.ylabel('Радиус корреляции, мкм')
plt.legend(frameon=0)
plt.tight_layout()
# plt.savefig('mmf/mmf_grin_cf_distance_exp.png', dpi=200)
plt.show()
