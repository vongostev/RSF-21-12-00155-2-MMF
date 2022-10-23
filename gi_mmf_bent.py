# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:34:32 2021

@author: vonGostev
"""
import __init__
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from lightprop2d import Beam2D, random_round_hole_phase, rectangle_hole, um, square_slits
from gi import GIEmulator
from gi.emulation import log, cached_ref_obj, low_res
from scipy.linalg import expm
from joblib import Parallel, delayed
from pathlib import Path

# Parameters
radius = 31.25  # in microns
n1 = 1.45
wl = 0.632  # wavelength in microns

# calculate the field on an area larger than the diameter of the fiber
area_size = 3.5 * radius
npoints = 2**8  # resolution of the window
xp = np
bounds = [-area_size / 2, area_size / 2]


def imshow(arr):
    plt.imshow(arr, extent=[-area_size / 2, area_size / 2] * 2)
    plt.xlabel(r'x, $\mu m$')
    plt.ylabel(r'y, $\mu m$')
    plt.show()

straight_path = [p for p in Path('data/').iterdir() if p.suffix == '.npz' and '0' in p.name][0]
with np.load(straight_path, allow_pickle=True) as data:
    fiber_op = data["fiber_op"]
    straight_modes_profiles = xp.array(data["modes_list"])
log.info(f"Straigt fiber params loaded")

fiber_len = 10 / um
straight_fiber_matrix = csc_matrix(expm(1j * fiber_op * fiber_len))
modes_matrix = xp.array(np.vstack(straight_modes_profiles).T)
straight_modes_matrix_t = modes_matrix.T
straight_modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)

def generate_beams(area_size, npoints, wl,
                   init_field, init_field_gen, init_gen_args,
                   object_gen, object_gen_args, object_init,
                   z_obj, z_ref, use_gpu, use_cupy, binning_order,
                   modes_profiles, modes_matrix_t, modes_matrix_dot_t,
                   fiber_matrix):

    if cached_ref_obj['ref'] is None:
        ref = Beam2D(area_size, npoints, wl,
                     init_field=init_field,
                     init_field_gen=init_field_gen,
                     init_gen_args=init_gen_args, use_gpu=use_gpu,
                     complex_bits=64,
                     numpy_output=not use_cupy)
        cached_ref_obj['ref'] = ref
    else:
        ref = cached_ref_obj['ref']
        ref.z = 0
        if init_field_gen is not None:
            field = init_field_gen(ref.X, ref.Y, *init_gen_args)
        if init_field is not None:
            field = init_field.copy()
        ref._update_obj(field)
    # modes_coeffs = ref.fast_deconstruct_by_modes(
    #     modes_matrix_t, modes_matrix_dot_t)
    # ref.construct_by_modes(modes_profiles, fiber_matrix @ modes_coeffs)

    if cached_ref_obj['obj'] is None:
        obj = Beam2D(area_size, npoints, wl, init_field=ref.field.copy(),
                     init_spectrum=ref.spectrum.copy(), use_gpu=use_gpu,
                     complex_bits=64,
                     numpy_output=not use_cupy)
        cached_ref_obj['obj'] = obj
    else:
        obj = cached_ref_obj['obj']
        obj.z = 0
        obj._update_obj(ref.field.copy(), ref.spectrum.copy())
    modes_coeffs = ref.fast_deconstruct_by_modes(
        straight_modes_matrix_t, straight_modes_matrix_dot_t)
    ref.construct_by_modes(straight_modes_profiles, straight_fiber_matrix @ modes_coeffs)
    modes_coeffs = obj.fast_deconstruct_by_modes(
        modes_matrix_t, modes_matrix_dot_t)
    obj.construct_by_modes(modes_profiles, fiber_matrix @ modes_coeffs)

    obj.propagate(z_obj)
    ref.propagate(z_ref)

    if object_gen is not None or object_init is not None:
        obj.coordinate_filter(
            f_gen=object_gen, fargs=object_gen_args, f_init=object_init)
    # plt.imshow(np.angle(ref.field.get()))
    # plt.show()
    # plt.imshow(obj.iprofile)
    # plt.show()
    return low_res(ref.iprofile, binning_order, obj.xp), low_res(obj.iprofile, binning_order, obj.xp)


def calc_gi(fiber_props, ifgen, nimgs):
    with np.load(fiber_props, allow_pickle=True) as data:
        fiber_op = data["fiber_op"]
        modes = xp.array(data["modes_list"])
    log.info(f"File `{fiber_props}` loaded")

    fiber_len = 10 / um
    fiber_matrix = csc_matrix(expm(1j * fiber_op * fiber_len))
    modes_matrix = xp.array(np.vstack(modes).T)
    modes_matrix_t = modes_matrix.T
    modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)

    emulator = GIEmulator(area_size*um, npoints,
                          wl*um, nimgs=nimgs,
                          init_field_gen=ifgen,
                          init_gen_args=(radius*um,),
                          iprofiles_gen=generate_beams,
                          iprofiles_gen_args=(
                              modes, modes_matrix_t,
                              modes_matrix_dot_t, fiber_matrix,
                          ),
                          object_gen=rectangle_hole,
                          object_gen_args=(20*um, 30*um),
                          use_gpu=1,
                          use_cupy=1,
                          binning_order=8
                          )

    emulator.calculate_ghostimage()
    # emulator.calculate_xycorr()
    print(emulator.ghost_data.max(), emulator.xycorr_data.max())
    return emulator.ghost_data#{'gi': emulator.ghost_data}


fiber_props_list = [p for p in Path('data/').iterdir() if p.suffix == '.npz' and '20' in p.name]
for p in fiber_props_list:
    print(p.name)

ifgen_list = [
    random_round_hole_phase,
]
params_keys = [p.name.split('_')[-1][:-4] for p in fiber_props_list]
print(params_keys)
params = np.array(np.meshgrid(fiber_props_list, ifgen_list)).reshape((2, -1)).T

nimgs = 1024
for k, p in zip(params_keys, params):
    _fiber_data = calc_gi(*p, nimgs)
    fname = f'mmf/gi_data_curv_{k}_straight.npy'
    log.info('Data saved to `{fname}`')
    np.save(fname, _fiber_data)