"""Model the behavioral data."""
# %%
import numpy as np
import pandas as pd

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_LOCAL, SUBJS
from utils import eq1, eq2, eq3, eq4, get_sourcedata

# %%
# Settings
streams = ["single", "dual"]

numbers = np.arange(1, 10)
numbers_rescaled = np.interp(numbers, (numbers.min(), numbers.max()), (-1, +1))


# %%
# Prepare file paths

analysis_dir = ANALYSIS_DIR_LOCAL
data_dir = DATA_DIR_LOCAL

# %%
# Get behavior data

for sub in SUBJS:
    for stream in streams:

        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "subject", sub)

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
# Rescale sample values to range [-1, 1]
# 1, 2, 3, 4, 5, 6, 7, 8, 9 --> -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
# X_rescaled = np.interp(X_abs, (X_abs.min(), X_abs.max()), (-1, +1))


def psychometric_model(X, categories, y, bias, kappa, leakage, noise, return_val):
    """Model the behavioral data as in Spitzer 2017, NHB [1]_.

    Parameters
    ----------
    X : np.ndarray, shape(300, 10)
        The data per participant and stream. Each row is a trial, each column
        is an unsigned sample value in the range [1, 9] that has been rescaled
        to the range [-1, 1].
    categories : np.ndarray, shape(300, 10)
        Signed array (-1, +1) of same shape as `X`. The sign represents each
        sample's color category (-1: red, +1: blue). For "single" stream data,
        this is an array of ones, in order to ignore the color category.
    y : np.ndarray, shape(300, 1)
        The information on which response was correct per trial. Can be ``0``
        or ``1`` (0: red/higher, 1: blue/lower; depending on stream single/dual).
    bias, kappa, leakage, noise : float
        The parameters of the model.
    return_val : {"neglog", "sse"}
        Whether to return negative log likelihood or sum of squared errors.

    Returns
    -------
    fit : float
        Either the "negative log likelihood" or the "sum of squared errors"
        of the model, depending on `return_val`.

    References
    ----------
    .. [1] Spitzer, B., Waschke, L. & Summerfield, C. Selective overweighting of larger
           magnitudes during noisy numerical comparison. Nat Hum Behav 1, 0145 (2017).
           https://doi.org/10.1038/s41562-017-0145

    See Also
    --------
    utils.eq1
    utils.eq2
    utils.eq3
    utils.eq4
    """
    # Transform sample values to subjective decision weights
    dv = eq1(X=X, bias=bias, kappa=kappa)

    # Obtain trial level decision variables
    DV = eq3(dv=dv, category=categories, gain=None, gnorm=False, leakage=leakage)

    # Convert decision variables to choice probabilities
    CP = eq4(DV, noise=noise)

    # Compute "model fit"
    if return_val == "neglog":
        fit = CP
    else:
        assert return_val == "sse"
        fit = CP

    return fit
