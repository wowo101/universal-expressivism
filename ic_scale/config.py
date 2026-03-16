"""Experiment parameters for IC/C scale invariance computation."""

import numpy as np

# Experiment parameters
L0 = 256
N_RG_LEVELS = 4        # levels 0–4: 256, 128, 64, 32, 16
M_VALUES = [2, 4]      # block sizes to measure
C_ADAPTIVE = 2.0       # adaptive-m parameter

# MC parameters
N_EQUIL = 5000         # Wolff steps for equilibration
N_SAMPLES = 5000       # independent configurations per temperature
THINNING = 5           # Wolff steps between samples

# Temperature grid
TC = 2.0 / np.log(1 + np.sqrt(2))
T_GRID = np.sort(np.concatenate([
    np.linspace(1.0, 1.8, 5),      # ordered phase (sparse)
    np.linspace(1.9, 2.7, 17),     # critical regime (dense)
    np.linspace(2.8, 4.0, 5),      # disordered phase (sparse)
]))

# MI parameters
ALPHA = 0.5            # Dirichlet regularisation
N_BLOCKS = 200         # block positions per sample
N_BOOTSTRAP = 50       # bootstrap resamples for error bars
