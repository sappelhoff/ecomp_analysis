"""Model the behavioral data."""
# %%
import itertools
import json
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import Bounds, minimize
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_LOCAL, NUMBERS, STREAMS, SUBJS
from utils import eq2, get_sourcedata, prep_model_inputs, psychometric_model

# %%
# Settings
numbers_rescaled = np.interp(NUMBERS, (NUMBERS.min(), NUMBERS.max()), (-1, +1))

param_names = ["bias", "kappa", "leakage", "noise"]

minimize_method = "Nelder-Mead"

# %%
# Prepare file paths

analysis_dir = ANALYSIS_DIR_LOCAL
data_dir = DATA_DIR_LOCAL

fname_estimates = analysis_dir / "derived_data" / f"estim_params_{minimize_method}.tsv"

# %%
# fit model
bias = 0
kappa = 1
leakage = 0
noise = 0.01
return_val = "G"

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
return_val = "G"

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
# Use a method that can work with bounds. "L-BFGS-B" is scipy default.
# "Nelder-Mead", "L-BFGS-B", "Powell" work
minimize_method_opts = {
    "Nelder-Mead": dict(maxiter=1000),
    "L-BFGS-B": dict(
        maxiter=1000, eps=1e-3
    ),  # https://stats.stackexchange.com/a/167199/148275
    "Powell": dict(
        maxiter=1000,
    ),
}[minimize_method]

# Initial parameter values
bias0 = 0
kappa0 = 1
leakage0 = 0
noise0 = 0.1

x0 = np.array([bias0, kappa0, leakage0, noise0])

if fname_estimates.exists():
    _df_prev = pd.read_csv(fname_estimates, sep="\t")

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

for sub in tqdm(SUBJS):
    for stream in STREAMS:

        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "subject", sub)

        X, categories, y, y_true, ambiguous = prep_model_inputs(df)

        # Add non-changing arguments to function
        kwargs = dict(
            X=X,
            categories=categories,
            y=y,
            return_val="G_noCP",
            gain=None,
            gnorm=False,
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

_df = pd.DataFrame.from_dict(data)
assert not np.any(~_df["success"])  # no failures

# Save the data
if fname_estimates.exists():
    pd.testing.assert_frame_equal(_df, _df_prev)
_df.to_csv(fname_estimates, sep="\t", na_rep="n/a", index=False)

# %%
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(1, 5, figsize=(10, 5))
    for iparam, param in enumerate(["bias", "kappa", "leakage", "noise", "loss"]):
        ax = axs.flat[iparam]

        sns.stripplot(x="stream", y=param, data=_df, ax=ax)
        sns.pointplot(x="stream", y=param, data=_df, ci=68, ax=ax, color="black")

        if param == "bias":
            ax.axhline(0, c="black", ls="--", lw=0.5)
        elif param == "kappa":
            ax.axhline(1, c="black", ls="--", lw=0.5)
        elif param == "leakage":
            ax.axhline(0, c="black", ls="--", lw=0.5)
        else:
            pass

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

sns.despine(fig)
fig.tight_layout()

# %%
# Run large set of (reasonable) initial start values per subj to find best ones

# Set reasonable bounds for the parameters (in param_names order)
lower = np.array([-1, 0, 0, 0.01], dtype=float)
upper = np.array([1, 5, 1, 5], dtype=float)
bounds = Bounds(lower, upper)

# Draw random initial values for the parameters
bias0s = np.arange(-5, 6) / 10
kappa0s = np.arange(0.2, 2.2, 0.2)
leakage0s = np.arange(0, 1.25, 0.25)
noise0s = np.arange(0.1, 1.1, 0.1)

x0s = list(itertools.product(bias0s, kappa0s, leakage0s, noise0s))

# Estimate parameters based on initial values for each dataset
# we save columns: sub-stream_idx-ix0-res.success-res.fun-x0-res.x
# = sub*streams*x0s rows
# takes about 125ms per fit, so (125*nrows)/1000 seconds overall
nrows = len(SUBJS) * len(STREAMS) * len(x0s)
secs = (125 * nrows) / 1000
print(f"Will run for about {secs} seconds ({secs/60/60:.2f}) hours.")
estimates = np.full(
    (len(x0s) * len(SUBJS) * len(STREAMS), 5 + len(param_names) * 2), np.nan
)
rowcount = 0
for sub in tqdm(SUBJS):
    for stream in STREAMS:
        for ix0, x0 in enumerate(x0s):
            _, tsv = get_sourcedata(sub, stream, data_dir)
            df = pd.read_csv(tsv, sep="\t")
            df.insert(0, "subject", sub)

            X, categories, y, y_true, ambiguous = prep_model_inputs(df)

            # Add non-changing arguments to function
            kwargs = dict(
                X=X,
                categories=categories,
                y=y,
                return_val="G_noCP",
                gain=None,
                gnorm=False,
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

            estimates[rowcount, ...] = np.array(
                [sub, STREAMS.index(stream), ix0, res.success, res.fun, *x0, *res.x]
            )
            rowcount += 1

# Save as npy
fname = str(fname_estimates).replace(".tsv", ".npy")
np.save(fname, estimates)

# turn into DataFrame
df_estimates = pd.DataFrame(
    estimates,
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

# got any failures?
not ~df_estimates["success"].to_numpy(dtype=bool).any()

# %%
# Fit all data as if from single subject
bias0s = (0, -0.1, 0.1)
kappa0s = (0.5, 1, 2)
leakage0s = (0, 0.2)
noise0s = (0.01, 0.1, 0.2)

x0s = []
for bias0, kappa0, leakage0, noise0 in itertools.product(
    bias0s, kappa0s, leakage0s, noise0s
):
    x0s.append(np.array([bias0, kappa0, leakage0, noise0]))

# boundaries for params (in order)
lower = np.array([-1, 0, 0, 0.01], dtype=float)
upper = np.array([1, 5, 1, 5], dtype=float)
bounds = Bounds(lower, upper)

# Collect all data as if from "single subject" (fixed effects)
df_single_sub = []
for sub in SUBJS:
    for istream, stream in enumerate(STREAMS):
        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "subject", sub)
        df_single_sub.append(df)

df_single_sub = pd.concat(df_single_sub)

# go through different initial values per stream
results = np.full((len(x0s), 2, 4), np.nan)  # init_vals X stream X params
for istream, stream in enumerate(STREAMS):

    X, categories, y, y_true, ambiguous = prep_model_inputs(
        df_single_sub[df_single_sub["stream"] == stream]
    )

    kwargs = dict(
        X=X,
        categories=categories,
        y=y,
        return_val="G_noCP",
        gain=None,
        gnorm=False,
    )

    fun = partial(psychometric_model, **kwargs)

    for ix0, x0 in enumerate(tqdm(x0s)):

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
# Turn results into DataFrame
_dfs = []
for ires in range(len(x0s)):
    _df = pd.DataFrame(results[ires, ...], columns=param_names)
    _df["stream"] = STREAMS
    _df["ix0s"] = ires
    _dfs.append(_df)
_df = pd.concat(_dfs)
_df = _df.melt(id_vars=["ix0s", "stream"], var_name="parameter")

# %%
# Plot results from single subj
with sns.plotting_context("talk"):
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.stripplot(x="parameter", y="value", hue="stream", data=_df, ax=ax, alpha=0.5)
    ax.axhline(1, ls="--", c="black", lw=0.5)
    ax.axhline(0, ls="--", c="black", lw=0.5)
    ax.legend(frameon=False)
    ax.set_title(
        "Estimation results\ndata combined over subjects ('fixed effects')\n"
        f"over {len(x0s)} different start parameters"
    )
sns.despine(fig)
fig.tight_layout()
# %%
