"""Model the behavioral data."""
# %%
# Imports
import itertools
import json
import sys
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
import scipy.stats
import seaborn as sns
import statsmodels.stats.multitest
from scipy.optimize import Bounds, minimize
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_LOCAL, NUMBERS, STREAMS, SUBJS
from utils import (
    eq2,
    find_dot_idxs,
    get_sourcedata,
    parse_overwrite,
    prep_model_inputs,
    psychometric_model,
)

# %%
# Settings
numbers_rescaled = np.interp(NUMBERS, (NUMBERS.min(), NUMBERS.max()), (-1, +1))

# Use a method that can work with bounds. "L-BFGS-B" is scipy default.
# "Nelder-Mead", "L-BFGS-B", "Powell" work, Nelder-Mead seems to work best.
minimize_method = "Nelder-Mead"
minimize_method_opts = {
    "Nelder-Mead": dict(maxiter=1000),
    "L-BFGS-B": dict(
        maxiter=1000, eps=1e-6
    ),  # https://stats.stackexchange.com/a/167199/148275
    "Powell": dict(
        maxiter=1000,
    ),
}[minimize_method]

analysis_dir = ANALYSIS_DIR_LOCAL
data_dir = DATA_DIR_LOCAL

overwrite = False

fit_scenario = "free"

fit_position = "all"  # "all", "firsthalf", "secondhalf", or "1", "2", ... "10"

do_plot = True

norm_leak = False

# see section below: "Fit all data as if from single subject"
do_fit_singlefx = False

# for plotting
SUBJ_LINE_SETTINGS = dict(color="black", alpha=0.1, linewidth=0.75)

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        analysis_dir=analysis_dir,
        data_dir=data_dir,
        overwrite=overwrite,
        do_plot=do_plot,
        fit_scenario=fit_scenario,
        fit_position=fit_position,
        norm_leak=norm_leak,
    )

    defaults = parse_overwrite(defaults)

    analysis_dir = defaults["analysis_dir"]
    data_dir = defaults["data_dir"]
    overwrite = defaults["overwrite"]
    do_plot = defaults["do_plot"]
    fit_scenario = defaults["fit_scenario"]
    fit_position = defaults["fit_position"]
    norm_leak = defaults["norm_leak"]

# %%
# Set parameter bounds (in order of `param_names`)
# and reasonable ranges for initial parameter values
# depending on `fit_scenario`
#
# fitting scenarios
# -----------------
# "free" -- the "standard case"
# "k_is_1" -- with k fixed to 1 (=)
# "k_bigger_1" -- with k not below 1 (>=)
# "k_smaller_1" -- with k not above 1 (<=)
# "leak" -- like "free", but with leakage parameter allowed
param_names = ["bias", "kappa", "leakage", "noise"]

if fit_scenario == "free":
    lower = np.array([-0.5, 0, 0, 0.01], dtype=float)
    upper = np.array([0.5, 5, 0, 3], dtype=float)
    kappa0s = np.arange(0.2, 2.2, 0.2)
    leakage0s = [0]
elif fit_scenario == "k_is_1":
    lower = np.array([-0.5, 1, 0, 0.01], dtype=float)
    upper = np.array([0.5, 1, 0, 3], dtype=float)
    kappa0s = [1]
    leakage0s = [0]
elif fit_scenario == "k_bigger_1":
    lower = np.array([-0.5, 1, 0, 0.01], dtype=float)
    upper = np.array([0.5, 5, 0, 3], dtype=float)
    kappa0s = np.arange(1.0, 2.2, 0.2)
    leakage0s = [0]
elif fit_scenario == "k_smaller_1":
    lower = np.array([-0.5, 0, 0, 0.01], dtype=float)
    upper = np.array([0.5, 1, 0, 3], dtype=float)
    kappa0s = np.arange(0.2, 1.0, 0.2)
    leakage0s = [0]
else:
    assert fit_scenario == "leak", f"unknown `fit_scenario`: {fit_scenario}"
    lower = np.array([-0.5, 0, -0.5, 0.01], dtype=float)
    upper = np.array([0.5, 5, 1, 3], dtype=float)
    kappa0s = np.arange(0.2, 2.2, 0.2)
    leakage0s = np.arange(-0.2, 0.4, 0.1)

bounds = Bounds(lower, upper)
bias0s = np.arange(-4, 5) / 10
noise0s = np.arange(0.1, 1.1, 0.1)

# number of free parameters for BIC and AIC calculation
n_free_params = len(param_names)
for lo, up in zip(lower, upper):
    if lo == up:
        n_free_params -= 1

# %%
# Prepare file paths
slug = f"_{fit_scenario}" if fit_scenario != "free" else ""
if slug == "_leak" and norm_leak:
    slug += "normed"

if fit_position == "all":
    fname_x0s = analysis_dir / "derived_data" / f"x0s_{minimize_method}{slug}.npy"
    fname_estimates = (
        analysis_dir / "derived_data" / f"estim_params_{minimize_method}{slug}.tsv"
    )
else:
    msg = (
        "fit_position must be 'all', 'firsthalf', 'secondhalf', or the string "
        "of a number between 1 and 10."
    )
    assert fit_position in [str(_) for _ in range(1, 11)] + [
        "firsthalf",
        "secondhalf",
    ], msg
    fname_x0s = (
        analysis_dir
        / "derived_data"
        / "fit_by_pos"
        / f"x0s_{minimize_method}{slug}_{fit_position}.npy"
    )
    fname_estimates = (
        analysis_dir
        / "derived_data"
        / "fit_by_pos"
        / f"estim_params_{minimize_method}{slug}_{fit_position}.tsv"
    )

fname_estimates.parent.mkdir(parents=True, exist_ok=True)
fname_x0s.parent.mkdir(parents=True, exist_ok=True)

fname_neurometrics = analysis_dir / "derived_data" / "neurometrics_params.tsv"
fname_neurometrics_erp = analysis_dir / "derived_data" / "erp_adm.tsv"
# %%
# Simulate accuracies over parameter ranges

# fixed model parameter values
bias = 0
kappa = 1
leakage = 0
noise = 0.1
return_val = "G"

# vary one parameter over these ranges while holding all others fixed
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

# NOTE: We run over subjects and streams, but results are almost identical as when
#       running over data from a single subject: The only variance comes from slightly
#       different underlying datasets. But this has a small influence given the uniform
#       distribution and large number of trials in these datasets.
for param, (data, xs_key, kwargs) in tqdm(simulation.items()):
    for x in xs_key:

        kwargs.update({param: x})
        parameters = np.array(
            [kwargs["bias"], kwargs["kappa"], kwargs["leakage"], kwargs["noise"]]
        )

        for sub in SUBJS:
            for stream in STREAMS:

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
                    norm_leak=norm_leak,
                )

                # Calculate accuracy on non-ambiguous, objectively correct choices
                acc = 1 - np.mean(np.abs(y_true[~ambiguous] - CP[~ambiguous]))

                # Save data
                data["subject"].append(sub)
                data["stream"].append(stream)
                data["accuracy"].append(acc)
                data[param].append(x)

# %%
# Plot accuracy simulation results
if do_plot:
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
# Simulate change in accuracy depending on noise and kappa parameters

# We can take data from any subj or stream, results will be nearly the same
sub = 32
stream = "dual"
_, tsv = get_sourcedata(sub, stream, data_dir)
df = pd.read_csv(tsv, sep="\t")
df.insert(0, "subject", sub)
X, categories, y, y_true, ambiguous = prep_model_inputs(df)

# Leave bias and leakage fixed at standard values
bias = 0
leakage = 0
return_val = "G"

# Vary kappa and noise
n = 101
kappas = np.linspace(0, 2.5, n)
noises = np.linspace(0.01, 2, n)[::-1]
idx_kappa_one = (np.abs(kappas - 1.0)).argmin()

# Apply different kinds of "gain normalization" to simulate limited-capacity agents
# (limited amount of "gain", e.g., firing speed of neurons, glucose in brain, ...)
gnorm_types = ["none", "experiment-wise", "trial-wise"]

# Collect data
acc_grid = np.full((n, n, len(gnorm_types)), np.nan)
for ignorm_type, gnorm_type in enumerate(tqdm(gnorm_types)):
    for ikappa, kappa in enumerate(kappas):

        # Setup gain normalization for this kappa parameterization
        gain = None
        gnorm = True
        if gnorm_type == "experiment-wise":
            feature_space = np.atleast_2d(numbers_rescaled)
        elif gnorm_type == "trial-wise":
            feature_space = X * categories
        else:
            assert gnorm_type == "none"
            gnorm = False

        if gnorm:
            gain = eq2(feature_space=feature_space, kappa=kappa, bias=bias)

        # Calculate accuracy for each noise level
        kwargs = dict(
            X=X,
            categories=categories,
            y=y,
            return_val=return_val,
            gain=gain,
            gnorm=gnorm,
            norm_leak=norm_leak,
        )
        for inoise, noise in enumerate(noises):
            parameters = np.array([bias, kappa, leakage, noise])
            _, CP = psychometric_model(parameters=parameters, **kwargs)

            acc = 1 - np.mean(np.abs(y_true[~ambiguous] - CP[~ambiguous]))

            acc_grid[inoise, ikappa, ignorm_type] = acc

# %%
# Plot "change in accuracy" simulations
if do_plot:
    with sns.plotting_context("talk", font_scale=1.4):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        for ignorm_type, gnorm_type in enumerate(gnorm_types):

            if ignorm_type == 1:
                continue
            elif ignorm_type == 2:
                ax = axs.flat[1]
            else:
                ax = axs.flat[ignorm_type]

            grid_norm = (
                acc_grid[..., ignorm_type].T - acc_grid[..., idx_kappa_one, ignorm_type]
            ).T

            # Trace maximum values using np.nan (inserts white cells)
            grid_norm[np.arange(n), np.argmax(grid_norm, axis=1)] = np.nan

            im = ax.imshow(
                grid_norm,
                origin="upper",
                interpolation="nearest",
                vmin=-0.11,
                vmax=0.025,
            )

            ax.axvline(idx_kappa_one, ls="--", c="w")

            if ignorm_type == 2:
                fig.colorbar(im, ax=ax, label="Δ Accuracy", shrink=1)

            # Set ticklabels
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            xticklabels = (
                [""]
                + [f"{i:.1f}" for i in kappas[(ax.get_xticks()[1:-1]).astype(int)]]
                + [""]
            )
            yticklabels = (
                [""]
                + [f"{i:.1f}" for i in noises[(ax.get_yticks()[1:-1]).astype(int)]]
                + [""]
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="FixedFormatter .* FixedLocator",
                )

                ax.set(
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                )

            ax.set(
                xlabel="Kappa (k)",
                ylabel="Noise (s)",
                title=f'Gain normalization:\n"{gnorm_type}"',
            )
            ax.set_ylabel(ax.get_ylabel(), labelpad=10)

        axs[0].set_title("Low task demands")
        axs[1].set_title("High task demands")

    fig.tight_layout()

    fname = analysis_dir / "figures" / "simul_plot.png"
    plt.savefig(fname, bbox_inches="tight", dpi=600)

# %%
# Fit model parameters for each subj and stream

# Initial guesses for parameter values: `x0`
bias0 = 0
kappa0 = 1
leakage0 = 0
noise0 = 0.1

x0 = np.array([bias0, kappa0, leakage0, noise0])

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

for sub in tqdm(SUBJS):
    for stream in STREAMS:

        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "subject", sub)

        X, categories, y, _, _ = prep_model_inputs(df)

        # Add non-changing arguments to function
        kwargs = dict(
            X=X,
            categories=categories,
            y=y,
            return_val="G_noCP",
            gain=None,
            gnorm=False,
            norm_leak=norm_leak,
        )
        fun = partial(psychometric_model, **kwargs)

        # estimate
        res = minimize(
            fun=fun,
            x0=x0,
            method=minimize_method,
            bounds=bounds,
            options=minimize_method_opts,
        )

        data["subject"].append(sub)
        data["stream"].append(stream)
        data["success"].append(res.success)
        data["loss"].append(res.fun)
        data["bias"].append(res.x[0])
        data["kappa"].append(res.x[1])
        data["leakage"].append(res.x[2])
        data["noise"].append(res.x[3])

df_fixed = pd.DataFrame.from_dict(data)

# Sanity check: no failures during fitting
assert not np.any(~df_fixed["success"])
df_fixed.drop(["success"], axis=1, inplace=True)

# This is data with "fixed" start values
df_fixed["bias0"] = bias0
df_fixed["kappa0"] = kappa0
df_fixed["leakage0"] = leakage0
df_fixed["noise0"] = noise0
df_fixed["x0_type"] = "fixed"
df_fixed["method"] = minimize_method

# Calculate and add AIC and BIC
# "loss" is based on G statistic, see `psychometric_model` function in `utils.py`
aic = 2 * n_free_params + df_fixed["loss"].to_numpy()

n_trials = 300  # per task
bic = np.log(n_trials) * n_free_params + df_fixed["loss"].to_numpy()

df_fixed.insert(df_fixed.columns.tolist().index("loss") + 1, "AIC", aic)
df_fixed.insert(df_fixed.columns.tolist().index("loss") + 1, "BIC", bic)


# %%
# Plot estimation results
def plot_estim_res(df, plot_single_subj, param_names):
    """Help to plot estimates."""
    hlines = dict(bias=0, kappa=1, leakage=0, noise=0)
    with sns.plotting_context("talk"):
        fig, axs = plt.subplots(1, len(param_names), figsize=(10, 5))
        for iparam, param in enumerate(param_names):
            ax = axs.flat[iparam]

            sns.pointplot(
                x="stream", y=param, data=df, order=STREAMS, ci=68, ax=ax, color="black"
            )
            if plot_single_subj:
                sns.swarmplot(
                    x="stream",
                    y=param,
                    data=df,
                    order=STREAMS,
                    ax=ax,
                    alpha=0.5,
                    size=1,
                )

                # https://stackoverflow.com/a/63171175/5201771
                set1 = df[df["stream"] == STREAMS[0]][param]
                set2 = df[df["stream"] == STREAMS[1]][param]

                _idx1, _idx2 = find_dot_idxs(ax, int(df_fixed.shape[0] / 2))
                locs1 = ax.get_children()[_idx1].get_offsets()
                locs2 = ax.get_children()[_idx2].get_offsets()

                sort_idxs1 = np.argsort(set1)
                sort_idxs2 = np.argsort(set2)

                locs2_sorted = locs2[sort_idxs2.argsort()][sort_idxs1]

                for i in range(locs1.shape[0]):
                    x = [locs1[i, 0], locs2_sorted[i, 0]]
                    y = [locs1[i, 1], locs2_sorted[i, 1]]
                    ax.plot(x, y, **SUBJ_LINE_SETTINGS)

            hline = hlines.get(param, None)
            if hline is not None:
                ax.axhline(hline, c="black", ls="--", lw=0.5)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_xlabel("")

    sns.despine(fig)
    fig.tight_layout()
    return fig, axs


if do_plot:
    fig, axs = plot_estim_res(df_fixed, plot_single_subj=True, param_names=param_names)
    fig.suptitle("Parameter estimates based on fixed initial values", y=1.05)

# %%
# Run large set of (reasonable) initial guesses per subj to find best ones
# NOTE: Depending on how many initial guesses to try, this will take a long time to run
#       ... could be sped up significantly through parallelization.
#
# Optionally run this separately for each sample position (1, 2, ..., 10), for example
# for checking recency/primacy, and whether parameters such as "k" vary over a trial
# see `fit_position`

if not fname_x0s.exists() or overwrite:

    x0s = list(itertools.product(bias0s, kappa0s, leakage0s, noise0s))

    # Estimate parameters based on initial values for each dataset
    # we save columns: sub,stream_idx,ix0,res.success,res.fun,x0,res.x
    # for `sub*streams*x0s` rows
    # script takes about 125ms per fit, so (125*nrows)/1000 seconds overall
    nrows = len(SUBJS) * len(STREAMS) * len(x0s)
    secs = (125 * nrows) / 1000
    print(f"Will run for about {secs} seconds ({secs/60/60:.2f}) hours.")
    x0_estimates = np.full(
        (len(x0s) * len(SUBJS) * len(STREAMS), 5 + len(param_names) * 2), np.nan
    )
    rowcount = 0
    for sub in tqdm(SUBJS):
        for stream in STREAMS:
            # get input values
            _, tsv = get_sourcedata(sub, stream, data_dir)
            df = pd.read_csv(tsv, sep="\t")
            df.insert(0, "subject", sub)

            X, categories, y, _, _ = prep_model_inputs(df)

            check_leakage = False
            if fit_position in ["firsthalf", "secondhalf"]:
                check_leakage = True
                if fit_position == "firsthalf":
                    X = X[:, 0:5]
                    categories = categories[:, 0:5]
                else:
                    assert fit_position == "secondhalf"
                    X = X[:, 5::]
                    categories = categories[:, 5::]
            elif fit_position != "all":
                check_leakage = True
                _pos = int(fit_position) - 1
                X = X[:, _pos, np.newaxis]
                categories = categories[:, _pos, np.newaxis]
            else:
                assert fit_position == "all"

            if check_leakage:
                msg = (
                    "for `fit_position` other than 'all', 'firsthalf', or "
                    "'secondhalf', leakage must be 0"
                )
                _leak_idx = param_names.index("leakage")
                assert lower[_leak_idx] == 0, msg
                assert upper[_leak_idx] == 0, msg
                assert leakage0s == [0], msg

            # Add non-changing arguments to function
            kwargs = dict(
                X=X,
                categories=categories,
                y=y,
                return_val="G_noCP",
                gain=None,
                gnorm=False,
                norm_leak=norm_leak,
            )
            fun = partial(psychometric_model, **kwargs)

            # Run different initial guesses
            for ix0, x0 in enumerate(x0s):
                res = minimize(
                    fun=fun,
                    x0=x0,
                    method=minimize_method,
                    bounds=bounds,
                    options=minimize_method_opts,
                )

                x0_estimates[rowcount, ...] = np.array(
                    [sub, STREAMS.index(stream), ix0, res.success, res.fun, *x0, *res.x]
                )
                rowcount += 1

    # Save as npy
    np.save(fname_x0s, x0_estimates)

else:
    # load if already saved
    print(f"\nInitial guesses x0 npy file already exists: {fname_x0s}\n\nLoading ...")
    x0_estimates = np.load(fname_x0s)

# turn into DataFrame and sanitize columns
df_x0s = pd.DataFrame(
    x0_estimates,
    columns=[
        "subject",
        "stream_idx",
        "ix0",
        "success",
        "loss",
        *[i + "0" for i in param_names],
        *param_names,
    ],
)

df_x0s = df_x0s.astype({"subject": int, "stream_idx": int, "ix0": int, "success": bool})
df_x0s["stream"] = df_x0s["stream_idx"].map(dict(zip(range(2), STREAMS)))

# drop failed estimations
nfail = np.sum(~df_x0s["success"].to_numpy())
nstartvals = len(df_x0s)
print(f"{(nfail/nstartvals)*100:.2f}% of fitting procedures failed.")
print("...selecting only successful fits\n")
df_x0s = df_x0s[df_x0s["success"].to_numpy()]

# Get the best fitting start values and estimates per subj and stream
df_specific = df_x0s.loc[df_x0s.groupby(["subject", "stream"])["loss"].idxmin()]
assert len(df_specific) == len(SUBJS) * len(STREAMS)
df_specific = df_specific[
    ["subject", "stream", "loss", *param_names, *[i + "0" for i in param_names]]
].reset_index(drop=True)
df_specific["x0_type"] = "specific"
df_specific["method"] = minimize_method

# Calculate and add AIC and BIC
# "loss" is based on G statistic, see `psychometric_model` function in `utils.py`
aic = 2 * n_free_params + df_specific["loss"].to_numpy()

n_trials = 300  # per task
bic = np.log(n_trials) * n_free_params + df_specific["loss"].to_numpy()

df_specific.insert(df_specific.columns.tolist().index("loss") + 1, "AIC", aic)
df_specific.insert(df_specific.columns.tolist().index("loss") + 1, "BIC", bic)

# if we fit by positions, save "specific" results
if fit_position != "all":
    df_specific.to_csv(fname_estimates, sep="\t", na_rep="n/a", index=False)

# %%
# Plot info on initial guesses
# plot distribution of "losses" per stream and subject,
# depending on start values
with sns.plotting_context("poster"):
    g = sns.catplot(
        kind="violin",
        data=df_x0s,
        x="stream",
        y="loss",
        col="subject",
        col_wrap=6,
    )

# %%
# plot distribution of best fitting initial start values
if do_plot:
    fig, axs = plot_estim_res(
        df_specific, plot_single_subj=True, param_names=[i + "0" for i in param_names]
    )
    _ = fig.suptitle(
        "Best fitting initial values over subjects\n"
        f"y-limits indicate ranges from which\n{df_x0s['ix0'].max()} "
        "initial values were tried out per subj and stream",
        y=1.15,
    )

# %%
# plot distribution of estimated params based on best fitting initial start values
if do_plot:
    fig, axs = plot_estim_res(
        df_specific, plot_single_subj=True, param_names=param_names
    )
    _ = fig.suptitle("Parameter estimates based on best fitting initial values", y=1.05)

# %%
# Work on stats for estimated params ("specific")
print("Means and standard errors:\n--------------------------")
for param in param_names:
    for stream in STREAMS:
        vals = df_specific[df_specific["stream"] == stream][param].to_numpy()
        m = np.mean(vals)
        se = scipy.stats.sem(vals)
        print(f"{param},{stream} --> {m:.2f} +- {se:.2f}")

# 1-samp tests vs "mu"
print("\n\n1-samp ttests vs mu\n-------------------")
use_one_sided = False
stats_params = []
_to_test = {"bias": 0, "kappa": 1, "leakage": 0}

# don't test parameters that we fixed
for ibound, (lo, up) in enumerate(zip(lower, upper)):
    if lo == up:
        del _to_test[param_names[ibound]]

for param, mu in _to_test.items():
    for stream in STREAMS:
        x = df_specific[df_specific["stream"] == stream][param].to_numpy()
        alt = "two-sided"
        if param == "kappa" and use_one_sided:
            alt = "greater" if stream == "dual" else "less"
        t, p = scipy.stats.wilcoxon(x - mu, alternative=alt)
        print(param, stream, np.round(p, 3), np.round(t, 3))
        pstats = pingouin.ttest(x, y=mu, alternative=alt)
        pstats["stream"] = stream
        pstats["parameter"] = param
        pstats["mu"] = mu
        stats_params.append(pstats)

        if param == "kappa":
            print(np.mean(x).round(2), np.mean(np.log(x)).round(2))
            pstats = pingouin.ttest(np.log(x), y=0, alternative=alt)
            pstats["stream"] = stream
            pstats["parameter"] = f"log({param})"
            pstats["mu"] = 0
            stats_params.append(pstats)

stats_params = pd.concat(stats_params).reset_index(drop=True)
print(
    "\n",
    stats_params[
        ["T", "dof", "alternative", "p-val", "cohen-d", "stream", "parameter", "mu"]
    ].round(3),
)

# paired test for noise
print("\n\npaired ttests noise\n------------------------------------------")
x = df_specific[df_specific["stream"] == "single"]["noise"].to_numpy()
y = df_specific[df_specific["stream"] == "dual"]["noise"].to_numpy()
stats_paired = pingouin.ttest(x, y, paired=True)
print("\n", stats_paired.round(3))

# %%
# Concatenate fixed and specific estimates and save
df_estimates = pd.concat([df_fixed, df_specific]).reset_index(drop=True)
assert len(df_estimates) == len(SUBJS) * len(STREAMS) * 2

# Save the data (except when we fit by positions ... then we saved earlier
# and without combining with fixed starting points)
if fit_position == "all":
    df_estimates.to_csv(fname_estimates, sep="\t", na_rep="n/a", index=False)

# %%
# Compare mean loss between estimates based on fixed vs. specific start values
if do_plot:
    with sns.plotting_context("talk"):
        g = sns.catplot(
            kind="point", x="x0_type", y="loss", col="stream", data=df_estimates, ci=68
        )

df_estimates.groupby(["stream", "x0_type"])["loss"].describe()

# %%
# Compare overall fit between fit_scenarios "free" and "leak"
estims_free = analysis_dir / "derived_data" / f"estim_params_{minimize_method}.tsv"
estims_leak = (
    analysis_dir
    / "derived_data"
    / f"estim_params_{minimize_method}_leak{'normed' if norm_leak else ''}.tsv"
)
if estims_free.exists() and estims_leak.exists():
    _free = pd.read_csv(estims_free, sep="\t")
    _leak = pd.read_csv(estims_leak, sep="\t")

    # compare overall fit by averaging over single/dual per subj
    _l = _leak.groupby(["subject", "x0_type"])[["AIC", "BIC"]].mean().reset_index()
    _l["fit_scenario"] = "leak"

    _f = _free.groupby(["subject", "x0_type"])[["AIC", "BIC"]].mean().reset_index()
    _f["fit_scenario"] = "free"

    df_fit_compare = pd.concat([_l, _f])

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    _all_stats = []
    for ax, goodness, x0_type in zip(
        axs.flat, ["AIC", "BIC"] * 2, ["fixed", "fixed", "specific", "specific"]
    ):

        # plot
        sns.pointplot(
            x="fit_scenario",
            y=goodness,
            order=["free", "leak"],
            data=df_fit_compare[df_fit_compare["x0_type"] == x0_type],
            ax=ax,
            ci=68,
        )
        ax.set_title(x0_type)

        # collect stats
        _ = df_fit_compare[df_fit_compare["x0_type"] == x0_type]
        x = _[_["fit_scenario"] == "free"][goodness]
        y = _[_["fit_scenario"] == "leak"][goodness]
        _stats = pingouin.ttest(x, y, paired=True)
        _stats["x0_type"] = x0_type
        _stats["goodness"] = goodness
        _all_stats.append(_stats)

    fig.tight_layout()
    sns.despine(fig)
    fig.suptitle("Normed leak" if norm_leak else "")

    # report stats
    _all_stats = pd.concat(_all_stats)
    _strmod = " (normed)" if norm_leak else ""
    print(f"\nGoodness of fit comparison between free and leak{_strmod}\n---")
    print(
        "\n",
        _all_stats[
            ["T", "dof", "alternative", "p-val", "cohen-d", "x0_type", "goodness"]
        ].round(3),
    )

else:
    print("skipping fit comparison between 'free' and 'leak'. Not all data present.")


# %%
# Correlation between noise and kappa per stream
with sns.plotting_context("talk"):
    g = sns.lmplot(
        x="noise",
        y="kappa",
        col_order=STREAMS,
        data=df_estimates,
        col="stream",
        row="x0_type",
    )

statsouts = []
for meta, grp in df_estimates.groupby(["x0_type", "stream"]):
    out = pingouin.corr(
        grp["noise"], grp["kappa"], method="pearson", alternative="two-sided"
    )
    out["x0_type"] = meta[0]
    out["stream"] = meta[1]
    statsouts.append(out)

statsout = pd.concat(statsouts).reset_index(drop=True)
print("\nCorrelation between noise and kappa per stream\n---")
print(
    "\n",
    statsout[["n", "r", "p-val", "x0_type", "stream"]].round(3),
)

# %%
# Correlate parameters within subjects (single vs dual)
_data = {"x0_type": [], "parameter": [], "single": [], "dual": []}
outs = []

# don't test parameters that we fixed
_to_test = []
for ibound, (lo, up) in enumerate(zip(lower, upper)):
    if lo != up:
        _to_test.append(param_names[ibound])

for x0_type in ["fixed", "specific"]:
    for param in _to_test:

        xy_list = []
        for stream in STREAMS:
            xy_list += [
                df_estimates[
                    (df_estimates["stream"] == stream)
                    & (df_estimates["x0_type"] == x0_type)
                ][param].to_numpy()
            ]

        x, y = xy_list
        out = pingouin.corr(x, y)
        out["x0_type"] = x0_type
        out["param"] = param
        outs.append(out)

        # save for plotting
        _data["x0_type"] += [x0_type] * len(SUBJS)
        _data["parameter"] += [param] * len(SUBJS)
        _data["single"] += x.tolist()
        _data["dual"] += y.tolist()

df_corrs = pd.concat(outs).reset_index(drop=True)

# plots
if do_plot:
    _data = pd.DataFrame.from_dict(_data)
    with sns.plotting_context("talk"):
        g = sns.lmplot(
            x="single",
            y="dual",
            col="parameter",
            row="x0_type",
            data=_data,
            facet_kws=dict(sharex=False, sharey=False),
        )

print("\nWithin-subject correlations: Single vs Dual\n---")
print(
    "\n",
    df_corrs[["n", "r", "p-val", "x0_type", "param"]].round(3),
)


# %%
# Correlate behavioral modelling and neurometrics "kappa" and "bias" parameters
# do for RSA neurometrics and ERP neurometrics
pvals_uncorr = {}
for neurom_type in ["rsa", "erp"]:
    if neurom_type == "rsa":
        fname = fname_neurometrics
    else:
        assert neurom_type == "erp"
        fname = fname_neurometrics_erp

    if not fname.exists():
        print(f"neurometrics params not found ... skipping.\n\n({fname})")
        continue
    else:
        print(f"{neurom_type}\n---")

    df_neurom = pd.read_csv(fname, sep="\t")

    # preprocess ERP data
    if neurom_type == "erp":
        df_neurom = df_neurom.drop_duplicates(["subject", "stream"]).reset_index(
            drop=True
        )[["subject", "stream", "bias", "kappa"]]

    if f"kappa_neuro_{neurom_type}" not in df_estimates.columns:
        df_estimates = df_estimates.merge(
            df_neurom,
            on=["subject", "stream"],
            suffixes=(None, f"_neuro_{neurom_type}"),
        )

    _df = df_estimates[
        [
            "subject",
            "stream",
            "bias",
            "kappa",
            f"bias_neuro_{neurom_type}",
            f"kappa_neuro_{neurom_type}",
            "x0_type",
        ]
    ]
    _df = _df.melt(id_vars=["subject", "stream", "x0_type"], var_name="parameter")

    x0_type = "specific"
    with sns.plotting_context("talk"):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for istream, stream in enumerate(STREAMS):
            for iparam, param in enumerate(["bias", "kappa"]):
                ax = axs[istream, iparam]

                x = _df[
                    (_df["stream"] == stream)
                    & (_df["x0_type"] == x0_type)
                    & (_df["parameter"] == param)
                ]["value"]
                y = _df[
                    (_df["stream"] == stream)
                    & (_df["x0_type"] == x0_type)
                    & (_df["parameter"] == f"{param}_neuro_{neurom_type}")
                ]["value"]
                assert len(x) == len(y)
                assert len(x) == len(SUBJS)
                ax.scatter(x, y)
                m, b = np.polyfit(x, y, 1)
                ax.plot(x, m * x + b, color="r")
                ax.set_title(f"{stream}")
                ax.set(xlabel=param, ylabel=param + f"_neuro_{neurom_type}")

                print(f"{stream}: {param} ~ {param}_neuro_{neurom_type}")
                # means
                print(f"means: {x.mean():.2f}, {y.mean():.2f}")

                # correlation
                r, p = scipy.stats.pearsonr(x, y)
                print(f"corr: r={r:.3f}, p={p:.3f}")
                pvals_uncorr[f"{neurom_type}-{stream}-{param}"] = p

                # paired t-test
                _stat = pingouin.ttest(x=x, y=y, paired=True)
                print(
                    f"paired ttest: t({_stat['dof'][0]})={_stat['T'][0]:.3f}, "
                    f"p={_stat['p-val'][0]:.3f}, d={_stat['cohen-d'][0]:.3f}\n"
                )

        sns.despine(fig)
        fig.tight_layout()

print("Multiple testing corrections of kappa correlations:")
keys = []
pvals_fdr = []
pvals_bonf = []
bonf_factor = sum([1 for i in list(pvals_uncorr.keys()) if "kappa" in i])
for key, val in pvals_uncorr.items():
    if "kappa" in key:
        keys.append(key)
        pvals_fdr.append(val)
        pvals_bonf.append(val * bonf_factor)

_, pvals_fdr = statsmodels.stats.multitest.fdrcorrection(pvals_fdr, alpha=0.05)

for key, pval_fdr, pval_bonf in zip(keys, pvals_fdr, pvals_bonf):
    print(
        f"{key}:\n    fdr: {np.round(pval_fdr, 3)}\n    bonf: {np.round(pval_bonf, 3)}"
    )

# %%
# Fit all data as if from single subject
_bias0s = (0, -0.1, 0.1)
_kappa0s = (0.5, 1, 2)
_leakage0s = (0, 0.2)
_noise0s = (0.01, 0.1, 0.2)

_x0s = []
for bias0, kappa0, leakage0, noise0 in itertools.product(
    _bias0s, _kappa0s, _leakage0s, _noise0s
):
    _x0s.append(np.array([bias0, kappa0, leakage0, noise0]))

# Collect all data as if from "single subject" (fixed effects)
df_single_sub = []
for sub in SUBJS:
    for istream, stream in enumerate(STREAMS):
        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "subject", sub)
        df_single_sub.append(df)

df_single_sub = pd.concat(df_single_sub).reset_index(drop=True)

# go through different initial values per stream
results = np.full((len(_x0s), len(STREAMS), len(param_names)), np.nan)
for istream, stream in enumerate(STREAMS):
    if not do_fit_singlefx:
        continue

    X, categories, y, _, _ = prep_model_inputs(
        df_single_sub[df_single_sub["stream"] == stream]
    )

    kwargs = dict(
        X=X,
        categories=categories,
        y=y,
        return_val="G_noCP",
        gain=None,
        gnorm=False,
        norm_leak=norm_leak,
    )

    fun = partial(psychometric_model, **kwargs)

    for ix0, x0 in enumerate(tqdm(_x0s)):

        res = minimize(
            fun=fun,
            x0=x0,
            method=minimize_method,
            bounds=bounds,
            options=minimize_method_opts,
        )

        assert res.success
        results[ix0, istream, ...] = res.x

# %%
# plot single subj "fixed effects" results
if not do_fit_singlefx:
    print("\nskipping 'fixed effects' analysis (`do_fit_singlefx` is False) ...")
else:
    # Turn results into DataFrame
    dfs_fixedfx = []
    for ires in range(len(_x0s)):
        df_fixedfx = pd.DataFrame(results[ires, ...], columns=param_names)
        df_fixedfx["stream"] = STREAMS
        df_fixedfx["ix0s"] = ires
        dfs_fixedfx.append(df_fixedfx)
    df_fixedfx = pd.concat(dfs_fixedfx).reset_index(drop=True)
    df_fixedfx = df_fixedfx.melt(id_vars=["ix0s", "stream"], var_name="parameter")

    # Plot results from single subj
    with sns.plotting_context("talk"):
        fig, ax = plt.subplots(figsize=(5, 6))
        sns.stripplot(
            x="parameter", y="value", hue="stream", data=df_fixedfx, ax=ax, alpha=0.5
        )
        ax.axhline(1, ls="--", c="black", lw=0.5)
        ax.axhline(0, ls="--", c="black", lw=0.5)
        ax.legend(frameon=False)
        ax.set_title(
            "Estimation results\ndata combined over subjects ('fixed effects')\n"
            f"over {len(_x0s)} different initial guesses"
        )
    sns.despine(fig)
    fig.tight_layout()
# %%
