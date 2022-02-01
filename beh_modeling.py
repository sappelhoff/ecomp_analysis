"""Model the behavioral data."""
# %%
import json
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import Bounds, minimize
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, CHOICE_MAP, DATA_DIR_LOCAL, SUBJS
from utils import eq1, eq2, eq3, eq4, get_sourcedata

# %%
# Settings
numbers = np.arange(1, 10)
numbers_rescaled = np.interp(numbers, (numbers.min(), numbers.max()), (-1, +1))

streams = ["single", "dual"]


# %%
# Prepare file paths

analysis_dir = ANALYSIS_DIR_LOCAL
data_dir = DATA_DIR_LOCAL

# %%
# Function to get behavior data


def prep_model_inputs(df):
    """Extract variables from behavioral data for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The behavioral data for a specific participant and
        task condition (stream).

    Returns
    -------
    X : np.ndarray, shape(n, 10)
        The data. Each row is a trial, each column is an unsigned sample value
        in the range [1, 9] that has been rescaled to the range [-1, 1]. In the
        eComp experiment, there were always 10 samples. The number of trials,
        `n` will be 300 in most cases but can be lower when a participant failed
        to respond for some trials (these trials are then dropped from analysis).
    categories : np.ndarray, shape(n, 10)
        Signed array (-1, +1) of same shape as `X`. For "dual" stream data, the
        sign represents each sample's color category (-1: red, +1: blue).
        For "single" stream data, this is an array of ones, in order to ignore
        the color category.
    y : np.ndarray, shape(n, 1)
        The participant's choices in this task condition (stream). Each entry
        correspondons to a trial and can be ``0`` or ``1``. In single stream
        condition, 0: "lower", 1: "higher". In dual stream condition:
        0: "red", 1: "blue".
    y_true : np.ndarray, shape(n, 1)
        The "objectively correct" choices per per trial. These are in the same
        format as `y`.
    ambiguous : np.ndarray, shape(n, 1)
        Boolean mask on which of the `n` valid trials contained samples that do
        not result in an objectively correct choice. For example in single stream,
        samples have a mean of exactly 5; or in dual stream, the red and blue
        samples have an identical mean.

    See Also
    --------
    config.CHOICE_MAP
    """
    # drop n/a trials (rows)
    idx_no_na = ~df["choice"].isna()

    # get categories
    sample_cols = [f"sample{i}" for i in range(1, 11)]
    X_signed = df.loc[idx_no_na, sample_cols].to_numpy()
    streams = df["stream"].unique()
    assert len(streams) == 1
    if streams[0] == "single":
        categories = np.ones_like(X_signed, dtype=int)
    else:
        assert streams[0] == "dual"
        categories = np.sign(X_signed)

    # Rescale sample values to range [-1, 1]
    # 1, 2, 3, 4, 5, 6, 7, 8, 9 --> -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
    X_abs = np.abs(X_signed)
    X = np.interp(X_abs, (X_abs.min(), X_abs.max()), (-1, +1))
    assert X_abs.min() == 1
    assert X_abs.max() == 9

    # map choices to 0 and 1
    y = df.loc[idx_no_na, "choice"].map(CHOICE_MAP).to_numpy()

    # Get "correct" choices. Single: 0=lower, 1=higher ; Dual: 0=red, 1=blue
    y_true = np.sign(np.sum((X * categories), axis=1)) / 2 + 0.5
    ambiguous = y_true == 0.5

    return X, categories, y, y_true, ambiguous


# %%
# Define model


def psychometric_model(
    parameters, X, categories, y, return_val, gain=None, gnorm=False
):
    """Model the behavioral data as in Spitzer 2017, NHB [1]_.

    Parameters
    ----------
    parameters : np.ndarray, shape(4,)
        An array containing the 4 parameters of the model::
            - bias : float
                The bias parameter (`b`) in range [-1, 1].
            - kappa : float
                The kappa parameter (`k`) in range [0, 20].
            -leakage : float
                The leakage parameter (`l`) in range [0, 1].
            - noise : float
                The noise parameter (`s`) in range [0.01, 8].
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
    return_val : {"neglog", "neglog_noCP", "sse"}
        Whether to return negative log likelihood or sum of squared errors.
        ``"neglog_noCP"`` returns the negative log likelihood *without*
        the additional `CP` return value.
    gain : np.ndarray, shape(n, 10) | None
        The gain normalization factor, where `n` is ``1`` if gain
        normalization is to be applied over the feature space of the
        entire experiment; or `n` is the number of trials if gain
        normalization is to be applied trial-wise.
        Can be ``None`` if `gnorm` is ``False``.
    gnorm : bool
        Whether to gain-normalize or not. Defaults to ``False``.

    Returns
    -------
    loss : float
        Either the "negative log likelihood" or the "sum of squared errors"
        of the model, depending on `return_val`.
    CP : np.ndarray, shape(n,)
        The probability to choose 1 instead of 0. One value per trial (`n` trials).
        Not returned if `return_val` is ``"neglog_noCP"``.


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
    # unpack parameters
    bias, kappa, leakage, noise = parameters

    # Transform sample values to subjective decision weights
    dv = eq1(X=X, bias=bias, kappa=kappa)

    # Obtain trial level decision variables
    DV = eq3(dv=dv, category=categories, gain=gain, gnorm=gnorm, leakage=leakage)

    # Convert decision variables to choice probabilities
    CP = eq4(DV, noise=noise)

    # Compute "model fit"
    if return_val in ["neglog", "neglog_noCP"]:
        # fix floating point issues to avoid log(0) = -inf
        # NOTE: These issues happen mostly when `noise` is very
        #       low (<0.1), and the subject chose an option opposite
        #       to the evidence. For example, noise=0.01, DV=0.5, y=0.
        #
        #       --> In this case, CP would be expit(x), where x is
        #           0.5/0.01 = 50, and expit(50) is 1.0, due to limitations
        #           in floating point precision.
        #       --> Next, when calculating the negative log likelihood
        #           for y == 0, we must do -np.log(1 - CP) and thus arrive
        #           at log(0), which evaluates to -inf.
        #
        #       The same can happen in the opposite case, where e.g.,
        #       noise=0.001, DV=-1.0, y=1.
        #
        #       --> Here, the corresponding expit(-1/0.001) is expit(-1000)
        #           and results in 0.
        #       --> Then, when calculating neg. log. lik. for y == 1, we
        #           must do -log(CP) and thus arrive at log(0) and
        #           -inf again.
        #
        #       We solve this issue by picking the floating point value closest
        #       to 1 (0.999999...) or 0 (0.00...01) instead of actually 1 or 0,
        #       whenever we run into this problem.
        #
        CP[(y == 1) & (CP == 0)] = np.nextafter(0.0, np.inf)
        CP[(y == 0) & (CP == 1)] = np.nextafter(1.0, -np.inf)
        loss = np.sum(-np.log(CP[y == 1])) + np.sum(-np.log(1.0 - CP[y == 0]))
        if return_val == "neglog_noCP":
            # for scipy.optimize.minimize, we must return a single float
            return loss
    else:
        assert return_val == "sse"
        loss = np.sum((y - CP) ** 2)

    return loss, CP


# %%
# fit model
bias = 0
kappa = 1
leakage = 0
noise = 0.01
return_val = "neglog"

simulation = {
    param: ({"subject": [], "stream": [], "accuracy": [], param: []}, xs, kwargs)
    for param, xs, kwargs in zip(
        ("bias", "kappa", "leakage", "noise"),
        (
            [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
            [0.1, 0.5, 1, 2, 4],
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            [0.01, 0.1, 1, 2, 4, 8],
        ),
        (
            {"kappa": kappa, "leakage": leakage, "noise": noise},
            {"bias": bias, "leakage": leakage, "noise": noise},
            {"bias": bias, "kappa": kappa, "noise": noise},
            {"bias": bias, "kappa": kappa, "leakage": leakage},
        ),
    )
}

for param, (data, xs_key, kwargs) in tqdm(simulation.items()):
    for x in xs_key:

        kwargs.update({param: x})
        parameters = np.array(
            [kwargs["bias"], kwargs["kappa"], kwargs["leakage"], kwargs["noise"]]
        )

        for sub in SUBJS:
            for stream in streams:

                _, tsv = get_sourcedata(sub, stream, data_dir)
                df = pd.read_csv(tsv, sep="\t")
                df.insert(0, "subject", sub)

                X, categories, y, y_true, ambiguous = prep_model_inputs(df)

                # Run model
                loss, CP = psychometric_model(
                    parameters=parameters,
                    X=X,
                    categories=categories,
                    y=y,
                    return_val=return_val,
                )

                # Calculate accuracy on non-ambiguous, objectively correct choices
                acc = 1 - np.mean(np.abs(y_true[~ambiguous] - CP[~ambiguous]))
                # print(loss, acc)

                # Save data
                data["subject"].append(sub)
                data["stream"].append(stream)
                data["accuracy"].append(acc)
                data[param].append(x)
# %%
# Plot simulation results
dfs = {}
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for i, param in enumerate(simulation):

        ax = axs.flat[i]

        # Get data and turn into df
        data, _, kwargs = simulation[param]
        kwargs.pop(param, None)
        df = pd.DataFrame.from_dict(data)
        dfs[param] = df

        # plot
        sns.pointplot(
            x=param,
            y="accuracy",
            hue="stream",
            data=df,
            ci=68,
            ax=ax,
            scale=0.5,
            dodge=True,
        )
        ax.set_title(json.dumps(kwargs)[1:-1])

        if i > 0:
            ax.get_legend().remove()

    fig.suptitle("Model run on participant data (N=30)", y=1.01)

sns.despine(fig)
fig.tight_layout()

# %%
# Run accuracy performance simulations

# We can take data from any subj or stream, results will be nearly the same
sub = 32
stream = "dual"
_, tsv = get_sourcedata(sub, stream, data_dir)
df = pd.read_csv(tsv, sep="\t")
df.insert(0, "subject", sub)
X, categories, y, y_true, ambiguous = prep_model_inputs(df)

# Leave bias and leakage at standard values
bias = 0
leakage = 0
return_val = "neglog"

# Vary kappa and noise
n = 101
kappas = np.linspace(0, 2.5, n)
noises = np.linspace(0.01, 2, n)[::-1]
gnorm_types = ["none", "experiment-wise", "trial-wise"]

idx_kappa_one = (np.abs(kappas - 1.0)).argmin()

# Collect data for different types of gain normalization
acc_grid = np.full((n, n, len(gnorm_types)), np.nan)
for ignorm_type, gnorm_type in enumerate(tqdm(gnorm_types)):
    for ikappa, kappa in enumerate(kappas):

        # Setup gain normalization for this kappa parameterization
        if gnorm_type == "experiment-wise":
            gain = eq2(
                feature_space=np.atleast_2d(numbers_rescaled), kappa=kappa, bias=bias
            )
            gnorm = True
        elif gnorm_type == "trial-wise":
            gain = eq2(feature_space=X * categories, kappa=kappa, bias=bias)
            gnorm = True
        else:
            assert gnorm_type == "none"
            gain = None
            gnorm = False

        # Calculate accuracy for each noise level
        kwargs = dict(
            X=X,
            categories=categories,
            y=y,
            return_val=return_val,
            gain=gain,
            gnorm=gnorm,
        )
        for inoise, noise in enumerate(noises):
            parameters = np.array([bias, kappa, leakage, noise])
            _, CP = psychometric_model(parameters=parameters, **kwargs)

            acc = 1 - np.mean(np.abs(y_true[~ambiguous] - CP[~ambiguous]))

            acc_grid[inoise, ikappa, ignorm_type] = acc

# %%
# Plot performance simulations
fig, axs = plt.subplots(3, 1, figsize=(5, 10))

for ignorm_type, gnorm_type in enumerate(gnorm_types):
    ax = axs.flat[ignorm_type]

    grid_norm = (
        acc_grid[..., ignorm_type].T - acc_grid[..., idx_kappa_one, ignorm_type]
    ).T

    # Trace maximum values using np.nan
    grid_norm[np.arange(n), np.argmax(grid_norm, axis=1)] = np.nan

    im = ax.imshow(grid_norm, origin="upper", interpolation="nearest")

    ax.axvline(idx_kappa_one, ls="--", c="w")
    fig.colorbar(im, ax=ax, label="Δ accuracy")

    # Set ticklabels
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    xticklabels = (
        [""] + [f"{i:.2f}" for i in kappas[(ax.get_xticks()[1:-1]).astype(int)]] + [""]
    )
    yticklabels = (
        [""] + [f"{i:.1f}" for i in noises[(ax.get_yticks()[1:-1]).astype(int)]] + [""]
    )

    ax.set(
        xlabel="curvature (k)",
        ylabel="noise (s)",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        title=f"Gain normalization:\n{gnorm_type}",
    )

fig.tight_layout()

# %%
# Try fitting parameters

# Initial parameter values
bias0 = 0
kappa0 = 1
leakage0 = 0
noise0 = 0.1

x0 = np.array([bias0, kappa0, leakage0, noise0])

# boundaries for params (in order)
lower = np.array([-1, 0, 0, 0.01], dtype=float)
upper = np.array([1, 5, 1, 5], dtype=float)
bounds = Bounds(lower, upper)

data = {
    "subject": [],
    "stream": [],
    "success": [],
    "loss": [],
    "bias": [],
    "kappa": [],
    "leakage": [],
    "noise": [],
}
for sub in SUBJS:
    for stream in streams:

        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "subject", sub)

        X, categories, y, y_true, ambiguous = prep_model_inputs(df)

        # Add non-changing arguments to function
        kwargs = dict(
            X=X,
            categories=categories,
            y=y,
            return_val="neglog_noCP",
            gain=None,
            gnorm=False,
        )
        fun = partial(psychometric_model, **kwargs)

        # estimate
        res = minimize(fun=fun, x0=x0, method="Nelder-Mead", bounds=bounds)

        data["subject"].append(sub)
        data["stream"].append(stream)
        data["success"].append(res.success)
        data["loss"].append(res.fun)
        data["bias"].append(res.x[0])
        data["kappa"].append(res.x[1])
        data["leakage"].append(res.x[2])
        data["noise"].append(res.x[3])

_df = pd.DataFrame.from_dict(data)
_df
# %%

fig, axs = plt.subplots(1, 4)
for iparam, param in enumerate(["bias", "kappa", "leakage", "noise"]):
    ax = axs.flat[iparam]

    sns.pointplot(data=_df, x="stream", y=param, ax=ax, ci=68)

fig.tight_layout()
# %%
# Save to check in matlab/octave
import scipy.io  # noqa: E402

f = "/home/stefanappelhoff/Downloads/dat.mat"
# b,Y,X,ML,nk,f,gnorm
dat = {
    "b": [0, bias, kappa, noise, leakage],
    "Y": y,
    "X": np.hstack([X, categories]),
    "ML": 1,
    "nk": 10,
    "f": np.arange(1, 11),
    "gnorm": False,
    "ambiguous": ambiguous,
    "y_true": y_true,
}
scipy.io.savemat(f, dat)

# %%
