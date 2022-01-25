"""Model the behavioral data."""
# %%
import numpy as np

from utils import eq1, eq2, eq3, eq4

# %%
numbers = np.arange(1, 10)
numbers_rescaled = np.interp(numbers, (numbers.min(), numbers.max()), (-1, +1))
# %%
bias = 0.5
kappa = 2
category = [-1, 1, 1, 1, 1, 1, 1, -1, -1]
gnorm = True
leakage = 0.5
nk = len(numbers_rescaled)
noise = 0.01

dv = eq1(numbers_rescaled, bias=bias, kappa=kappa)
gain = eq2(numbers_rescaled, bias=bias, kappa=kappa)
DV = eq3(dv, category, gain, gnorm, leakage, seq_length=nk)
CP = eq4(DV, noise)
for i in (dv, gain, DV, CP):
    print(i)

# %%
