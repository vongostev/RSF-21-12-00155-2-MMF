# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:02:32 2022

@author: Pavel Gostev
"""

import __init__
from datetime import date
from tqdm import tqdm
from pyMMF import (IndexProfile, propagationModeSolver,
                   estimateNumModesGRIN, estimateNumModesSI)
from lightprop2d import (
    Beam2D, random_round_hole_phase, random_round_hole, um,
    plane_wave, random_wave_bin, random_round_hole_bin)
from gi.experiment import corr1d3d
from dataclasses import dataclass, field
from scipy.sparse.linalg import expm
from scipy.sparse import csc_matrix
from time import perf_counter
import numpy as np
import cupy as cp

from loguru import logger as log
from typing import Union

"""
1. Распространение случайного поля в волокне и за ним при бинарной амплитудной модуляции
2. Распространение случайного поля в волокне и за ним при амплитудной модуляции
3. Распространение случайного поля в волокне и за ним при фазовой модуляции
4. Распространение случайного поля в волокне и за ним при амплитудно-фазовой модуляции
    - Картинка профиля до волокна
    - Картинка профиля на входе в волокно (собранного из мод)
    - Картинка поля на выходе из волокна и на разных расстояниях от него
    - Корреляционная функция исходного поля и корреляционная функция поля на
    входе в волокно
    - Корреляционная функция на выходе волокна и на разных расстояниях от него

"""

def _np(data: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
    """Convert cupy or numpy arrays to numpy array.

    Parameters
    ----------
    data : Union[numpy.ndarray, cupy.ndarray]
        Input data.

    Returns
    -------
    data : numpy.ndarray
        Converted data.

    """
    # Return numpy array from numpy or cupy array
    if type(data).__module__ == 'cupy':
        return data.get()
    return data


def fiber_data_gen(obj, fiber_len, prop_distance):
    obj.init_beam()
    _iprofile = obj.get_output_iprofile(fiber_len)
    if prop_distance > 0:
        obj.propagate(prop_distance)
        return obj.prop_iprofile
    else:
        return _iprofile


@dataclass
class LightFiberAnalyser:
    
    xp: object = np

    area_size: float = 40
    npoints: int = 2**8
    wl: float = 0.632
    init_field_gen: object = plane_wave
    init_gen_args: tuple = ()

    NA: float = 0.2
    n1: float = 1.45
    core_radius: float = 15

    simc__kwargs: dict = \
        field(
            default_factory=lambda: dict(
                core_pitch=5,
                dims=1,
                layers=1,
                delta=0.039,
                central_core=True)
        )
    pcf__kwargs: dict = \
        field(
            default_factory=lambda: dict(
                core_pitch=5,
                dims=1,
                layers=1,
                delta=0.039,
                cladding_radius=25)
        )

    index_type: str = 'GRIN'
    fiber_type: str = 'mmf'
    curvature: float = None
    solver_mode: str = 'eig'
    fiber_len: float = 0

    use_gpu: bool = True
    nimg: int = 1000

    mod_radius: float = 0

    def __post_init__(self):
        
        if self.use_gpu:
            self.xp = cp

        profile = self.__set_index_profile()
        self.solver = propagationModeSolver()
        self.solver.setIndexProfile(profile)
        self.solver.setWL(self.wl)

        self.fiber_props = \
            f"data/{self.fiber_type}_{self.index_type}_{self.core_radius * 2:g}_properties_{int(self.curvature * um) if self.curvature else 0}.npz"
        self.__o__fields = []

    def __set_index_profile(self):
        profile = IndexProfile(npoints=self.npoints,
                               areaSize=self.area_size)
        if self.index_type == 'GRIN':
            profile.initParabolicGRIN(
                n1=self.n1,
                a=self.core_radius,
                NA=self.NA)
            self.Nmodes_estim = estimateNumModesGRIN(
                self.wl,
                self.core_radius,
                self.NA,
                pola=1)

        elif self.index_type == 'SI':
            profile.initStepIndex(
                n1=self.n1,
                a=self.core_radius,
                NA=self.NA)
            self.Nmodes_estim = estimateNumModesSI(
                self.wl,
                self.core_radius,
                self.NA, pola=1)

        elif self.index_type == 'SIMC':
            profile.initStepIndexMultiCoreRadial(
                n1=self.n1,
                a=self.core_radius,
                NA=self.NA,
                **self.simc__kwargs)
            self.Nmodes_estim = estimateNumModesSI(
                self.wl,
                self.core_radius,
                self.NA,
                pola=1) * (
                    self.simc__kwargs['dims'] * self.simc__kwargs['layers'])

        elif self.index_type == 'PCF':
            self.Nmodes_estim = 990
            profile.initPhotonicCrystalHex(
                n1=self.n1,
                a=self.core_radius,
                NA=self.NA,
                **self.pcf__kwargs)

        log.success('Index profile has been calculateed')
        return profile

    def set_modulation_func(self, mod_func, *args, **kwargs):
        self._mf = lambda x, y: mod_func(x, y, *args, **kwargs)

    def fiber_calc(self):
        t = perf_counter()
        try:
            with np.load(self.fiber_props, allow_pickle=True) as data:
                self.fiber_op = data["fiber_op"]
                self.modes = self.xp.asarray(data["modes_list"])
                self.betas = data["betas"]
            log.info(f'Fiber data loaded from `{self.fiber_props}`')
            
        except FileNotFoundError:
            log.warning(f"Fiber data not found in `{self.fiber_props}`")
            log.info("Starting calculations...")
            if self.index_type == 'SI' and not self.curvature:
                modes = self.solver.solve(mode='SI', n_jobs=-2)
            else:
                modes = self.solver.solve(
                    nmodesMax=self.Nmodes_estim+100, boundary='close',
                    mode='eig', curvature=self.curvature, propag_only=True)

            self.betas = self.xp.asarray(modes.betas)
            self.modes = self.xp.asarray(modes.profiles)[
                self.xp.argsort(self.betas)[::-1]]
            self.betas = self.xp.sort(self.betas)[::-1]
            self.fiber_op = csc_matrix(modes.getEvolutionOperator())

            np.savez_compressed(self.fiber_props,
                                fiber_op=self.fiber_op,
                                modes_list=_np(self.modes),
                                betas=_np(self.betas),
                                allow_pickle=True)
            log.info(f'Fiber data saved to `{self.fiber_props}`')

        modes_matrix = self.xp.array(self.xp.real(self.xp.vstack(self.modes).T))
        self.modes_matrix_t = self.xp.array(modes_matrix.T)
        self.modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)
        log.info(f'Found {len(self.modes)} modes')
        log.success(
            f"Fiber initialized. Elapsed time {perf_counter() - t:.3f} s")
        self.modes_coeffs = self.xp.zeros((self.nimg, len(self.modes)),
                                     dtype=np.complex128)

    def init_beam(self, field=None):
        if field is None:
            self.beam = Beam2D(self.area_size, self.npoints, self.wl,
                               init_field_gen=self.init_field_gen,
                               init_gen_args=self.init_gen_args,
                               use_gpu=self.use_gpu,
                               numpy_output=False)
            self.mask = self._mf(self.beam.X, self.beam.Y)
            self.beam.coordinate_filter(f_init=self.mask)
        else:
            self.beam = Beam2D(self.area_size, self.npoints, self.wl,
                               init_field=field,
                               use_gpu=self.use_gpu,
                               numpy_output=False)

    def propagate(self, z=0):
        self.beam.propagate(z)

    def get_index_profile(self):
        return self.solver.indexProfile.n.reshape([self.npoints] * 2)

    def set_transmission_matrix(self, fiber_len):
        t = perf_counter()
        self.tm = self.xp.array(expm(1j * self.fiber_op * fiber_len).todense())
        log.info(
            f'Transmission matrix calculated. Elapsed time {perf_counter() - t} s')

    @property
    def iprofile(self):
        return self.beam.iprofile

    def get_input_iprofile(self):
        return self.get_output_iprofile(0)

    def get_output_iprofile(self, fiber_len=0, mc=None):
        if mc is None:
            mc = self.beam.fast_deconstruct_by_modes(
                self.modes_matrix_t, self.modes_matrix_dot_t)
        if fiber_len > 0:
            mc = self.tm @ self.beam._asxp(mc)
        self.beam.construct_by_modes(self.modes, mc)
        return self.iprofile

    def _get_cf(self, obj_data, ref_data, parallel_njobs=-1, fast=False):
        t = perf_counter()
        self.cf = corr1d3d(obj_data, ref_data)
        log.info(
            f"Correlation function calculated. Elapsed time {perf_counter() - t:.3f} s")

    def correlate_init(self, nimg=0):
        t = perf_counter()
        idata = self.xp.zeros((nimg, self.npoints, self.npoints))
        for i in range(nimg):
            self.init_beam()
            idata[i, :, :] = self.iprofile
        point_data = idata[:, self.npoints // 2, self.npoints // 2]
        log.info(
            f"Initial data to cf generated. Elapsed time {perf_counter() - t:.3f} s")
        self._get_cf(point_data, idata)

        return self.cf

    def correlate_input(self, nimg=0):
        nimg = self.nimg if nimg == 0 else nimg
        t = perf_counter()
        idata = self.xp.zeros((nimg, self.npoints, self.npoints))
        for i in tqdm(range(nimg), position=0, leave=True):
            self.init_beam()
            _iprofile = self.get_output_iprofile(0)
            idata[i, :, :] = _iprofile
            self.modes_coeffs[i, :] = self.beam.modes_coeffs
        point_data = idata[:, self.npoints // 2, self.npoints // 2]
        log.info(
            f"In-fiber data to cf generated. Elapsed time {perf_counter() - t:.3f} s")
        self._get_cf(point_data, idata)
        return self.cf

    def correlate_output(self, nimg=0, fiber_len=0, prop_distance=0, expand=1):
        nimg = self.nimg if nimg == 0 else nimg
        t = perf_counter()
        idata = self.xp.zeros((nimg, self.npoints, self.npoints))
        __fields = []
        for i in tqdm(range(nimg), position=0, leave=True):
            if len(self.__o__fields) == nimg and expand > 1:
                self.init_beam()
                self.init_beam(
                    self.__o__fields[i])
                _iprofile = self.beam.iprofile
            else:
                self.init_beam()
                _iprofile = self.get_output_iprofile(
                    fiber_len, self.modes_coeffs[i])
            if prop_distance > 0:
                if expand > 1:
                    self.beam.expand(self.area_size * expand)
                    self.beam.coarse(expand)
                self.propagate(prop_distance / um)
                idata[i, :, :] = self.beam.iprofile
            else:
                idata[i, :, :] = _iprofile
            __fields.append(self.beam.field)

        self.__o__fields = __fields
        self.area_size = self.area_size * expand
        point_data = idata[:, self.npoints // 2, self.npoints // 2]
        log.info(
            f"In-fiber data to cf generated. Elapsed time {perf_counter() - t:.3f} s")
        self._get_cf(point_data, idata)
        return self.cf

    def correlate_by_fiber_len(self, nimg=0, max_fiber_len=1 / um):
        data = []
        for l in np.linspace(0, max_fiber_len, 15):
            self.set_transmission_matrix(l)
            data.append(self.correlate_output(nimg, l, 0))
        return data


fiber_params = [
    dict(
        core_radius=4.5,
        npoints=256,
        area_size=3.5 * 4.5,  # um
        # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=358
        index_type='SI',
        NA=0.22,
        # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
        n1=1.4613,
        mod_radius=4.5,
        curvature=10 / um
    ),
    # dict(
    #     npoints=256,
    #     area_size=3.5 * 31.25,  # um
    #     # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=358
    #     index_type='GRIN',
    #     core_radius=31.25,
    #     NA=0.275,
    #     # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
    #     n1=1.4613,
    #     mod_radius=31.25,
    #     curvature=5 / um
    # ),
    # dict(
    #     npoints=256,
    #     area_size=3.5 * 31.25,  # um
    #     # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=6838
    #     index_type='SI',
    #     core_radius=25,
    #     NA=0.22,
    #     # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
    #     n1=1.4613,
    #     mod_radius=25
    # ),
    # dict(
    #     area_size=4 * 3,  # um
    #     # to SIMC IXF-MC-12-PAS-6
    #     index_type='SI',
    #     fiber_type='smf_simc',
    #     core_radius=3,
    #     NA=0.19,
    #     # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
    #     n1=1.4613,
    #     mod_radius=3
    # ),
    # dict(
    #     npoints=128,
    #     area_size=12 * 1.5,  # um
    #     # https://www.thorlabs.com/drawings/b2e64c24c4214c42-DB6047F0-B8E1-ED20-62AA1F597ADBE2AB/S405-XP-SpecSheet.pdf
    #     index_type='SI',
    #     fiber_type='smf',
    #     core_radius=1.5,
    #     NA=0.12,
    #     # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
    #     n1=1.4613,
    #     mod_radius=1.5
    # ),
    # dict(
    #     area_size=3.5 * 31.25,
    #     npoints=256,
    #     # https://photonics.ixblue.com/sites/default/files/2021-06/IXF-MC-12-PAS-6_edA_multicore_fiber.pdf
    #     index_type='SIMC',
    #     core_radius=3,
    #     NA=0.19,
    #     n1=1.4613,
    #     simc__kwargs=dict(
    #         core_pitch=35,
    #         delta=0.039,
    #         dims=12,
    #         layers=1,
    #         central_core_radius=0
    #     ),
    #     mod_radius=42
    # ),
    # dict(
    #     # https://www.thorlabs.com/thorproduct.cfm?partnumber=S405-XP
    #     area_size=150,
    #     npoints=400,
    #     index_type='PCF',
    #     core_radius=1.8,
    #     NA=0.2,
    #     n1=1.4613,
    #     pcf__kwargs=dict(
    #         central_core_radius=5,
    #         central_core_n1=1,
    #         core_pitch=0.2,
    #         pcf_radius=37,
    #         cladding_radius=60
    #     ),
    #     mod_radius=60
    # )
]

fiber_data = {}

mod_params = {
    # 'dmd': {
    #     'init_gen': plane_wave,
    #     'init_args': (),
    #     'mod_gen': random_round_hole_bin
    # },
    # 'ampl': {
    #     'init_gen': plane_wave,
    #     'init_args': (),
    #     'mod_gen': random_round_hole
    # },
    'slm': {
        'init_gen': plane_wave,
        'init_args': (),
        'mod_gen': random_round_hole_phase
    },
    # 'dmdslm': {
    #     'init_gen': random_wave_bin,
    #     'init_args': (),
    #     'mod_gen': random_round_hole_phase
    # },
}

# um
fiber_len = 10 / um  # um for cm
real_distances = []# [0, 20, 40, 60, 80, 100, 150, 200, 400, 1000, 3000, 10000]
distances = np.diff(np.array([0] + real_distances)) * um
expands = [1] #* 6 + [2] * 1 + [1] * 2 + [2] * 3
n_cf = 1000

_dtoday = date.today()
_date = f'{_dtoday.day:02d}{_dtoday.month:02d}{_dtoday.year % 2000:02d}'

data_dir = 'mmf'
max_flen = 23 / um
use_gpu = True
PREFIX = 'cohdata_dist'

if use_gpu:
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

for mod in mod_params:
    for params in fiber_params:
        itype = params['index_type']
        log.info(
            f"Analysing {itype} fiber with diameter {params['core_radius'] * 2} um")
        fiber_data[itype] = {'params': params}
        mod_radius = params['mod_radius']

        analyser = LightFiberAnalyser(
            use_gpu=use_gpu,
            init_field_gen=mod_params[mod]['init_gen'],
            init_gen_args=mod_params[mod]['init_args'],
            nimg=n_cf,
            **params)
        analyser.set_modulation_func(mod_params[mod]['mod_gen'],
                                     mod_radius, binning_order=1)
        analyser.init_beam()
        # Пример исходного профиля
        fiber_data[itype]['s__ip'] = _np(analyser.iprofile)
        # Исходная корреляционная функция
        fiber_data[itype]['s__cf'] = _np(analyser.correlate_init(n_cf))

        analyser.fiber_calc()
        # Профиль показателя преломления
        fiber_data[itype]['index'] = _np(analyser.get_index_profile())
        # Пример профиля на входе волокна после разложения по модам
        fiber_data[itype]['i__ip'] = _np(analyser.get_input_iprofile())
        # Корреляционная функция на входе волокна
        fiber_data[itype]['i__cf'] = _np(analyser.correlate_input(n_cf))

        analyser.set_transmission_matrix(fiber_len)
        log.info(f'Set fiber length to {fiber_len * um} cm')
        for expf, d, rd in zip(expands, distances, real_distances):
            log.info(f"Set propagation distance to {d:g} cm")
            # Корреляционная функция после волокна на расстоянии d см
            fiber_data[itype][f'o__cf_{rd}'] = _np(analyser.correlate_output(
                n_cf, fiber_len, prop_distance=d, expand=expf))
            # Пример профиля после волокна на расстоянии d см
            fiber_data[itype][f'o__ip_{rd}'] = _np(analyser.iprofile)
        # fiber_data[itype]['params']['max_flen'] = max_flen
        # fiber_data[itype]['fl__cf'] = analyser.correlate_by_fiber_len(
        #     n_cf, max_fiber_len=max_flen)

    fname = f'{data_dir}/{PREFIX}_{_date}_{analyser.fiber_type}_{analyser.core_radius * 2:.2g}_{mod}_{int(analyser.curvature * um) if analyser.curvature else 0}.npz'
    np.savez_compressed(fname, **fiber_data)
    log.info(f'Data saved to `{fname}`')

    if use_gpu:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
