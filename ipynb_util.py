import __init__
import numpy as np
import cupy as cp
from pathlib import Path
import matplotlib.pyplot as plt
from lightprop2d import Beam2D
from cupyx.scipy.sparse import csc_matrix
from scipy.linalg import expm


def plot_modes(modes, indexes, bounds, curv):
    npoints = np.ceil(np.sqrt(len(modes[0]))).astype(int)
    indexes_map = np.arange(len(modes))[indexes]
    nrows = len(indexes) // 5
    fig, axes = plt.subplots(nrows, 5, dpi=100, figsize=(10, nrows * 2 + 1))
    for modnum, mode, ax in zip(indexes_map, modes[indexes], axes.flatten()):
        ax.imshow(np.abs(mode.get()).reshape(
            (npoints, -1)) ** 2, extent=bounds * 2)
        ax.set_title(f'Мода №{modnum}')
    # plt.savefig('pcf/pcf_modes.png', dpi=100)
    # plt.savefig('simc/smf_simc_modes.png', dpi=100)
    fig.suptitle(
        f'Радиус кривизны оптического волокна {curv if curv else "∞"} см', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_profiles_by_curv(profiles):
    fig, ax = plt.subplots(1, len(profiles), figsize=(10, 4))
    for i, c in enumerate(profiles):
        ax[i].imshow(profiles[c])
        ax[i].set_title(f'{c if c else "∞"} см')

    plt.tight_layout()
    plt.show()


def calc_inout(modes, op, ref, fiber_len, max_modenum=None):
    if max_modenum is None:
        idx = len(modes)
    else:
        idx = max_modenum
    test = Beam2D(ref.area_size, ref.npoints, ref.wl, init_field=ref.field.copy(),
                  init_spectrum=ref.spectrum.copy(), use_gpu=True,
                  complex_bits=64)
    _modes = modes[:idx]
    # modes_matrix = np.vstack(_modes).T
    # modes_matrix_t = modes_matrix.T
    # modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)
    fiber_matrix = csc_matrix(expm(1j * op * fiber_len))[:idx, :idx]

    # m = test.fast_deconstruct_by_modes(
    #     modes_matrix_t, modes_matrix_dot_t)
    m = test.deconstruct_by_modes(_modes)
    test.construct_by_modes(_modes, m)
    res = [test.field, test.spectrum]
    m1 = fiber_matrix @ m
    test.construct_by_modes(_modes, m1)
    res += [test.field, test.spectrum, m, m1]
    return res


def plot_inout_data(data, c):
    f, s, f1, s1, _, _ = data
    f = f.get()
    f1 = f1.get()
    s = np.fft.fftshift(s.get())
    s1 = np.fft.fftshift(s1.get())
    fig, ax = plt.subplots(2, 4, figsize=(12, 5))
    data_in = {
        'Амплитуда поля на входе': np.abs(f),
        'Фаза поля на входе': np.angle(f),
        'Спектр. амплитуда на входе': np.abs(s),
        'Спектр. фаза на входе': np.angle(s)
    }
    data_out = {
        'Амплитуда поля на выходе': np.abs(f1),
        'Фаза поля на выходе': np.angle(f1),
        'Спектр. амплитуда на выходе': np.abs(s1),
        'Спектр. фаза на выходе': np.angle(s1)
    }
    for i, (lbl, d) in enumerate(data_in.items()):
        im = ax[0, i].imshow(d)
        ax[0, i].set_title(lbl)
        fig.colorbar(im, ax=ax[0, i])
    for i, (lbl, d) in enumerate(data_out.items()):
        im = ax[1, i].imshow(d)
        ax[1, i].set_title(lbl)
        fig.colorbar(im, ax=ax[1, i])
    fig.suptitle(
        f'Радиус кривизны оптического волокна {c if c > 0 else "∞"} см', fontsize=16)
    plt.tight_layout()
    plt.show()


def load_fiber_data(mask, dir='data'):
    files = sorted(Path(dir).glob(mask))
    modes = {}
    ops = {}
    betas = {}
    for f in files:
        c = int(f.name.split('_')[-1][:-4])
        d = np.load(f, mmap_mode='r', allow_pickle=True)
        betas[c] = d['betas']
        modes[c] = cp.asarray(d['modes_list'][np.argsort(betas[c])[::-1]])
        betas[c] = sorted(betas[c])[::-1]
        ops[c] = d['fiber_op']

    return modes, betas, ops


def ssim_lum(x, y):
    mx = x.mean()
    my = y.mean()
    return (2 * mx * my + 1e-8) / (mx ** 2 + my ** 2 + 1e-8)


def ssim_contrast(x, y):
    sx = x.std()
    sy = y.std()
    return (2 * sx * sy + 1e-8) / (sx ** 2 + sy ** 2 + 1e-8)


def ssim_struct(x, y):
    sx = x.std()
    sy = y.std()
    covxy = (x * y).mean() - x.mean() * y.mean()
    return (covxy + 5e-9) / (sx * sy + 5e-9)


def SSIM(x, y, a=1, b=1, c=1):
    x = x / x.max()
    y = y / y.max()
    x = (x * 255).astype(np.uint8) / 255
    y = (y * 255).astype(np.uint8) / 255
    return ssim_lum(x, y) ** a * ssim_contrast(x, y) ** b * ssim_struct(x, y) ** c
