# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 23:46:48 2022

@author: vonGostev
"""


import numpy as np
import matplotlib.pyplot as plt



fig, axes = plt.subplots(2, 5, dpi=100, figsize=(10, 4))
for mode, ax in zip(modes, axes.flatten()):
    ax.imshow(np.real(mode).reshape((256, -1)),
              extent=bounds * 2)
plt.tight_layout()
# plt.savefig('pcf/pcf_modes.png', dpi=100)
# plt.savefig('simc/smf_simc_modes.png', dpi=100)
plt.show()