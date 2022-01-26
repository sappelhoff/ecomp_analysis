"""Model the behavioral data."""
# %%
import numpy as np
import pandas as pd

from config import ANALYSIS_DIR_LOCAL, CHOICE_MAP, DATA_DIR_LOCAL, SUBJS
from utils import eq1, eq3, eq4, get_sourcedata

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


def prep_model_inputs(df):
    """Extract variables from behavioral data for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The behavioral data.

    Returns
    -------
    X : np.ndarray, shape(n, 10)
        The data per participant and stream. Each row is a trial, each column
        is an unsigned sample value in the range [1, 9] that has been rescaled
        to the range [-1, 1]. In the eComp experiment, there were always
        10 samples. The number of trials, `n` will be 300 in most cases but can
        be lower when a participant failed to respond for some trials (these
        trials are then dropped from analysis).
    categories : np.ndarray, shape(n, 10)
        Signed array (-1, +1) of same shape as `X`. The sign represents each
        sample's color category (-1: red, +1: blue). For "single" stream data,
        this is an array of ones, in order to ignore the color category.
    y : np.ndarray, shape(n, 1)
        The choices per participant and stream. Each entry is the choice on a
        given trial. Can be ``0`` or ``1``. In single stream condition,
        0: "lower", 1: "higher". In dual stream condition: 0: "red", 1: "blue".
    """
    # drop n/a trials (rows)
    idx_no_na = ~df["choice"].isna()

    # get categories
    sample_cols = [f"sample{i}" for i in range(1, 11)]
    X_signed = df.loc[idx_no_na, sample_cols].to_numpy()
    X_abs = np.abs(X_signed)
    streams = df["stream"].unique()
    assert len(streams) == 1
    if streams[0] == "single":
        categories = np.ones_like(X_abs, dtype=int)
    else:
        assert streams[0] == "dual"
        categories = np.sign(X_abs)

    # Rescale sample values to range [-1, 1]
    # 1, 2, 3, 4, 5, 6, 7, 8, 9 --> -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
    X = np.interp(X_abs, (X_abs.min(), X_abs.max()), (-1, +1))
    assert X_abs.min() == 1
    assert X_abs.max() == 9

    # map choices to 0 and 1
    y = df.loc[idx_no_na, "choice"].map(CHOICE_MAP).to_numpy()

    return X, categories, y


for sub in SUBJS:
    for stream in streams:

        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "subject", sub)

        X, categories, y = prep_model_inputs(df)

# %%


def psychometric_model(X, categories, y, bias, kappa, leakage, noise, return_val):
    """Model the behavioral data as in Spitzer 2017, NHB [1]_.

    Parameters
    ----------
    X : np.ndarray, shape(n, 10)
        The data per participant and stream. Each row is a trial, each column
        is an unsigned sample value in the range [1, 9] that has been rescaled
        to the range [-1, 1]. In the eComp experiment, there were always
        10 samples. The number of trials, `n` will be 300 in most cases but can
        be lower when a participant failed to respond for some trials (these
        trials are then dropped from analysis).
    categories : np.ndarray, shape(n, 10)
        Signed array (-1, +1) of same shape as `X`. The sign represents each
        sample's color category (-1: red, +1: blue). For "single" stream data,
        this is an array of ones, in order to ignore the color category.
    y : np.ndarray, shape(n, 1)
        The choices per participant and stream. Each entry is the choice on a
        given trial. Can be ``0`` or ``1``. In single stream condition,
        0: "lower", 1: "higher". In dual stream condition: 0: "red", 1: "blue".
    bias : float
        The bias parameter (`b`) in range [-1, 1].
    kappa : float
        The kappa parameter (`k`) in range [0, np.inf].
    leakage : float
        The leakage parameter (`l`) in range [0, 1].
    noise : float
        The noise parameter (`s`) in range [1e-10, np.inf].
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
    config.CHOICE_MAP
    """
    # Transform sample values to subjective decision weights
    dv = eq1(X=X, bias=bias, kappa=kappa)

    # Obtain trial level decision variables
    DV = eq3(dv=dv, category=categories, gain=None, gnorm=False, leakage=leakage)

    # Convert decision variables to choice probabilities
    CP = eq4(DV, noise=noise)

    # Compute "model fit"
    if return_val == "neglog":
        # fix floating point issues to avoid log(0) = inf
        CP[(y == 1) & (CP == 0)] = np.nextafter(0.0, np.inf)
        CP[(y == 0) & (CP == 1)] = np.nextafter(1.0, -np.inf)

        fit = np.sum(-np.log(CP[y == 1])) + np.sum(-np.log(1.0 - CP[y == 0]))
    else:
        assert return_val == "sse"
        fit = np.sum((y - CP) ** 2)

    return fit, CP, DV, dv


# %%
# fit model
bias = 0.2
kappa = 1
leakage = 0
noise = 0.4
return_val = "neglog"
fit, CP, DV, dv = psychometric_model(
    X, categories, y, bias, kappa, leakage, noise, return_val
)
fit
# %%
CP.min()
# %%
CP.max()

# %%
