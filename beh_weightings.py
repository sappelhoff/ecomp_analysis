"""Analyze weights."""
# %%
# Imports
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
import seaborn as sns
from scipy.stats import sem

from config import (
    ANALYSIS_DIR_LOCAL,
    CHOICE_MAP,
    DATA_DIR_LOCAL,
    NUMBERS,
    STREAMS,
    SUBJS,
)
from utils import (
    get_estim_params,
    get_sourcedata,
    parse_overwrite,
    prep_model_inputs,
    prep_weight_calc,
    psychometric_model,
)

# %%
# Settings
analysis_dir = ANALYSIS_DIR_LOCAL
data_dir = DATA_DIR_LOCAL

positions = np.arange(10)

minimize_method = "Nelder-Mead"

x0_type = "specific"

# may be: "free" (default), "k_is_1", "k_smaller_1", "k_bigger_1"
fit_scenario = "free"

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        analysis_dir=analysis_dir,
        data_dir=data_dir,
        fit_scenario=fit_scenario,
    )

    defaults = parse_overwrite(defaults)

    analysis_dir = defaults["analysis_dir"]
    data_dir = defaults["data_dir"]
    fit_scenario = defaults["fit_scenario"]

# %%
# Prepare file paths
slug = f"_{fit_scenario}" if fit_scenario != "free" else ""
fname_estimates = (
    analysis_dir / "derived_data" / f"estim_params_{minimize_method}{slug}.tsv"
)

# %%
# Define function to calculate weights


def calc_nonp_weights(df, nsamples=10):
    """Calculate non-parametric weights.

    In the "single stream" task, the weight of a number is its
    relative frequency with which it led to a "larger" choice.
    In the "dual stream" task, the weight of a number is its
    relative frequency with which its color was selected as
    a choice.

    Parameters
    ----------
    df : pandas.DataFrame
        The behavioral data of a subject.
    nsamples : int
        Number of samples in each trial. Defaults to 10,
        as in the eComp experiment.

    Returns
    -------
    weights : np.ndarray, shape(9,)
        The weight for each of the 9 numbers in ascending
        order (1 to 9).
    position_weights : np.ndarray, shape(10, 9)
        The weight for each of the 9 numbers in ascending
        order, calculated for each of the 10 sample positions.

    See Also
    --------
    calc_CP_weights
    """
    weights_df, stream, samples, colors, positions = prep_weight_calc(df, nsamples)

    if stream == "single":
        # sanity check
        np.testing.assert_array_equal(
            np.unique(weights_df["choice"]), ["higher", "lower"]
        )

        # map lower/higher to 0/1 ... and repeat choice for each sample
        choices = np.repeat(
            weights_df["choice"].map(CHOICE_MAP).to_numpy(),
            nsamples,
        )
        assert choices.shape == samples.shape
    else:
        assert stream == "dual"
        # sanity check
        np.testing.assert_array_equal(np.unique(weights_df["choice"]), ["blue", "red"])

        # map red/blue to 0/1 ... and repeat choice for each sample
        choices = np.repeat(weights_df["choice"].map(CHOICE_MAP).to_numpy(), nsamples)
        assert choices.shape == samples.shape

    # Calculate overall weights and for each sample position
    numbers = np.arange(1, 10, dtype=int)  # numbers 1 to 9 were shown
    weights = np.zeros(len(numbers))
    position_weights = np.zeros((nsamples, len(numbers)))
    for inumber, number in enumerate(numbers):

        if stream == "single":

            # overall weights
            weights[inumber] = np.mean(choices[samples == number])

            # weights for each sample position
            for pos in np.unique(positions):
                position_weights[pos, inumber] = np.mean(
                    choices[(samples == number) & (positions == pos)]
                )

        else:
            assert stream == "dual"

            # overall weights
            weights[inumber] = np.mean(
                np.hstack(
                    [
                        choices[(samples == number) & (colors == 1)],
                        -choices[(samples == number) & (colors == 0)] + 1,
                    ]
                )
            )

            # weights for each sample position
            for pos in np.unique(positions):
                position_weights[pos, inumber] = np.mean(
                    np.hstack(
                        [
                            choices[
                                (samples == number) & (colors == 1) & (positions == pos)
                            ],
                            -choices[
                                (samples == number) & (colors == 0) & (positions == pos)
                            ]
                            + 1,
                        ]
                    )
                )

    return weights, position_weights


def calc_CP_weights(
    sub, stream, x0_type, data_dir, analysis_dir, minimize_method, slug, params=[]
):
    """Calculate decision weights based on model output `CP`.

    Each weight is the mean of its associated CP values.

    Parameters
    ----------
    sub : int
        Which subject the data should be based on.
    stream : {"single", "dual"}
        The stream this data should be based on.
    x0_type : {"fixed", "specific"}
        Whether to work on estimated parameters based on fixed initial guesses
        or subject/stream-specific initial guesses.
    data_dir : pathlib.Path
        The path to the data directory.
    analysis_dir : pathlib.Path
        The path to the analysis directory.
    minimize_method : {"Neldar-Mead", "L-BFGS-B", "Powell"}
        The method with which the parameters were estimated.
    slug : {"", "k_is_1", "k_bigger_1", "k_smaller_1"}
        Modify the file name of the estimated parameters.
    params : list
        Can be an empty list to use the subject and stream specific
        estimated parameters as read from file (default).
        Can be a list of four floats corresponding to bias, kappa,
        leakage, and noise in that order. If one of the four list
        entries is None, the estimated parameters as read from file
        are used instead.
        Can be helpful to fit CP_weights based on mean estimated params.
        Or to plot model outputs at kappa=1.

    Returns
    -------
    weights : np.ndarray, shape(9,)
        The weight for each of the 9 numbers in ascending
        order (1 to 9).
    position_weights : np.ndarray, shape(10, 9)
        The weight for each of the 9 numbers in ascending
        order, calculated for each of the 10 sample positions.

    See Also
    --------
    calc_nonp_weights
    """
    # Get data for modeling
    _, tsv = get_sourcedata(sub, stream, data_dir)
    df = pd.read_csv(tsv, sep="\t")
    df.insert(0, "subject", sub)

    X, categories, y, y_true, ambiguous = prep_model_inputs(df)

    # Get estimated parameters and predicted choices
    parameters = get_estim_params(
        sub, stream, x0_type, minimize_method, slug, analysis_dir
    )
    if params != []:
        # replace some or all parameters
        assert len(params) == 4
        none_idxs = [i for i, p in enumerate(params) if p is None]
        for idx in none_idxs:
            params[idx] = parameters.tolist()[idx]
        parameters = np.asarray(params, dtype=float)

    _, CP = psychometric_model(
        parameters, X, categories, y, return_val="G", gain=None, gnorm=False
    )

    # prepare weighting data
    nsamples = 10
    weights_df, _, samples, colors, positions = prep_weight_calc(df, nsamples)

    # repeat CP for each sample
    CPs = np.repeat(CP, nsamples)

    # Calculate weights
    numbers = np.arange(1, 10, dtype=int)
    weights = np.zeros(len(numbers))
    position_weights = np.zeros((nsamples, len(numbers)))
    for inumber, number in enumerate(numbers):

        # overall weights
        if stream == "single":
            weights[inumber] = np.mean(CPs[samples == number])
        else:
            assert stream == "dual"
            weights[inumber] = np.mean(
                np.hstack(
                    [
                        CPs[(samples == number) & (colors == 1)],
                        (1 - CPs[(samples == number) & (colors == 0)]),
                    ]
                )
            )

        # weights for each sample position
        for pos in np.unique(positions):

            if stream == "single":
                position_weights[pos, inumber] = np.mean(
                    CPs[(samples == number) & (positions == pos)]
                )
            else:
                assert stream == "dual"
                position_weights[pos, inumber] = np.mean(
                    np.hstack(
                        [
                            CPs[
                                (samples == number) & (colors == 1) & (positions == pos)
                            ],
                            (
                                1
                                - CPs[
                                    (samples == number)
                                    & (colors == 0)
                                    & (positions == pos)
                                ]
                            ),
                        ]
                    )
                )

    return weights, position_weights


# %%
# calculate weights over subjects

# Take model parameter estimates based on "fixed" or "specific" initial guess (x0_type)
model_kwargs = dict(
    x0_type=x0_type,
    data_dir=data_dir,
    analysis_dir=analysis_dir,
    minimize_method=minimize_method,
    slug=slug,
)
wtypes = ["data", "model", "model_k1"]
weight_dfs = []
posweight_dfs = []
for sub in SUBJS:
    for stream in STREAMS:

        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")

        for wtype in wtypes:
            if wtype == "data":
                weights, position_weights = calc_nonp_weights(df)
            elif wtype == "model":
                weights, position_weights = calc_CP_weights(
                    sub,
                    stream,
                    **model_kwargs,
                )
            elif wtype == "model_k1":
                weights, position_weights = calc_CP_weights(
                    sub, stream, **model_kwargs, params=[None, 1, None, None]
                )
            else:
                raise RuntimeError("unrecognized `wtype`")

            # save in DF
            wdf = pd.DataFrame.from_dict(
                dict(
                    subject=sub,
                    stream=stream,
                    weight_type=wtype,
                    number=NUMBERS,
                    weight=weights,
                    x0_type=x0_type,
                )
            )
            pwdf = pd.DataFrame.from_dict(
                dict(
                    subject=sub,
                    stream=stream,
                    weight_type=wtype,
                    position=np.repeat(positions, len(NUMBERS)),
                    number=np.tile(NUMBERS, len(positions)),
                    weight=position_weights.flatten(),  # shape(pos X weight)
                    x0_type=x0_type,
                )
            )

            weight_dfs.append(wdf)
            posweight_dfs.append(pwdf)

weightdata = pd.concat(weight_dfs).reset_index(drop=True)
posweightdata = pd.concat(posweight_dfs).reset_index(drop=True)

# %%
# Model-free analysis of compression vs. anti-compression
# we take the weight difference of mean(1-2, 8-9) and mean(4-5, 5-6)
# a positive value means compression: the slopes at the edges
# are shallower than in the middle
# and vice versa: a negative value means anti-compression

for stream in STREAMS:
    datas = []
    for num in [1, 2, 8, 9, 4, 5, 6]:
        datas.append(
            weightdata[
                (weightdata["stream"] == stream)
                & (weightdata["weight_type"] == "data")
                & (weightdata["number"] == num)
            ]["weight"].to_numpy()
        )

    datas = np.stack(datas)

    vals = ((datas[0, :] - datas[1, :]) + (datas[2, :] - datas[3, :])) / 2 - (
        (datas[4, :] - datas[5, :]) + (datas[5, :] - datas[6, :])
    ) / 2

    # ttest against 0
    _stats = pingouin.ttest(vals, y=0)
    t = _stats["T"].to_numpy()[0]
    dof = _stats["dof"].to_numpy()[0]
    p = _stats["p-val"].to_numpy()[0]
    d = _stats["cohen-d"].to_numpy()[0]

    print(
        f"{stream}: t({dof})={t:.2f}, p={p:.3f}, d={d:.2f}, two-tailed;"
        f" mean: {np.mean(vals):.3f}, SEM: {sem(vals):.3f}"
    )


# %%
# plot weights
plotkwargs = dict(
    x="number",
    y="weight",
    data=weightdata,
    dodge=False,
    ci=68,
)
plotgrid = True
if plotgrid:
    g = sns.catplot(
        hue="weight_type",
        col="stream",
        kind="point",
        **plotkwargs,
    )
else:
    fig, ax = plt.subplots()
    sns.pointplot(
        hue="stream",
        **plotkwargs,
    )
    ax.axhline(0.5, linestyle="--", color="black", lw=0.5)

# plot regression lines per task
plot_reg_lines = True
if plot_reg_lines and not plotgrid:
    _tmp = weightdata.groupby(["stream", "number"])["weight"].mean().reset_index()
    for _stream, _color in zip(STREAMS, ["C0", "C1"]):
        xy = _tmp[_tmp["stream"] == _stream][["number", "weight"]].to_numpy()
        m, b = np.polyfit(xy[:, 0], xy[:, 1], 1)
        ax.plot(np.arange(9), m * xy[:, 0] + b, color=_color)

# optional horizontal, vertical, and diagonal (=linear weights) reference lines
fname = analysis_dir / "figures" / f"weights{slug}.jpg"
plot_ref_lines = False
refline_kwargs = dict(linestyle="--", color="black", lw=0.5)
if plot_ref_lines and not plotgrid:
    ax.axhline(0.5, **refline_kwargs)
    ax.axvline(4, **refline_kwargs)
    fname = str(fname).replace(".jpg", "_reflines.jpg")
if plot_ref_lines and plotgrid:
    for col_val, ax in g.axes_dict.items():
        ax.axhline(0.5, **refline_kwargs)
        ax.axvline(4, **refline_kwargs)

if plotgrid:
    sns.despine(g.fig)
    g.fig.suptitle(
        f"Model based on initial guesses of type '{x0_type}' {slug})", y=1.05
    )
    g.fig.savefig(fname)
else:
    sns.despine(fig)
    fig.savefig(fname)

# %%
# Plot single subj fits
df_estimates = pd.read_csv(fname_estimates, sep="\t")

for stream in STREAMS:
    with sns.plotting_context("poster"):
        g = sns.catplot(
            x="number",
            y="weight",
            data=weightdata[weightdata["stream"] == stream],
            dodge=False,
            ci=68,
            hue="weight_type",
            hue_order=["data", "model", "model_k1"],
            col="subject",
            col_wrap=6,
            kind="point",
        )
    _ = g.fig.suptitle(f'"{stream} stream" task {slug}', y=1.05, fontsize=40)

    for isub, (col_val, ax) in enumerate(g.axes_dict.items()):
        _df = df_estimates[
            (df_estimates["x0_type"] == x0_type)
            & (df_estimates["subject"] == col_val)
            & (df_estimates["stream"] == stream)
        ]
        b, k = _df[["bias", "kappa"]].to_numpy().flatten()
        title = ax.get_title()
        ax.set_title(f"sub: {isub+1}\nb: {b:.2f}, k: {k:.2f}", fontsize=30)
        ax.axhline(0.5, linestyle="--", color="black", lw=0.5)

    g.fig.tight_layout()
    with sns.plotting_context("poster", font_scale=1.5):
        sns.move_legend(obj=g, loc="upper right", ncol=3, title="")

# %%
# Plot fit based on mean estimates parameters
param_names = ["bias", "kappa", "leakage", "noise"]
_df_mean = df_estimates.groupby(["x0_type", "stream"])[param_names].mean().reset_index()


weight_dfs = []
for sub in SUBJS:
    for stream in STREAMS:

        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")

        params = (
            _df_mean[(_df_mean["x0_type"] == x0_type) & (_df_mean["stream"] == stream)][
                param_names
            ]
            .to_numpy()
            .squeeze()
            .tolist()
        )

        weights, _ = calc_CP_weights(
            sub,
            stream,
            x0_type,
            data_dir,
            analysis_dir,
            minimize_method,
            slug,
            params=params,
        )

        # save in DF
        wdf = pd.DataFrame.from_dict(
            dict(
                subject=sub,
                stream=stream,
                weight_type="model_mean",
                number=NUMBERS,
                weight=weights,
                x0_type=x0_type,
            )
        )
        weight_dfs.append(wdf)
weightdata_mean = pd.concat(weight_dfs).reset_index(drop=True)
if "model_mean" not in weightdata["weight_type"]:
    weightdata = pd.concat([weightdata, weightdata_mean]).reset_index(drop=True)


# %%
# plot weights over positions: numbers as hue
g = sns.catplot(
    x="position",
    y="weight",
    hue="number",
    data=posweightdata,
    dodge=False,
    ci=68,
    palette="crest_r",
    col="stream",
    row="weight_type",
    kind="point",
)

g.fig.suptitle(f"Model based on initial guesses of type '{x0_type}' {slug}", y=1.05)
fname = analysis_dir / "figures" / f"posweights_numberhue{slug}.jpg"
g.fig.savefig(fname)

# %%
# plot over positions: data and model overlaid
palette = "muted"  # "crest_r"

with sns.plotting_context("talk"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

    for istream, stream in enumerate(STREAMS):

        data = posweightdata[(posweightdata["stream"] == stream)]

        ax = axs.flat[istream]

        # plot data
        sns.pointplot(
            x="position",
            y="weight",
            hue="number",
            data=data[data["weight_type"] == "data"],
            ci=68,
            palette=palette,
            ax=ax,
        )

        # plot model
        _ = (
            data[data["weight_type"] == "model"]
            .groupby(["position", "number"])
            .mean()
            .reset_index()[["position", "number", "weight"]]
        )
        _

        for inumber, number in enumerate(NUMBERS):
            color = sns.color_palette(palette, n_colors=len(NUMBERS))[inumber]
            ax.plot(
                _[_["number"] == number]["position"].to_numpy(),
                _[_["number"] == number]["weight"].to_numpy(),
                marker="o",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=5,
                zorder=10,
                color="black",
                lw=0.5,
            )

        handles, labels = ax.get_legend_handles_labels()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="You have mixed positional .*"
            )
            sns.move_legend(
                ax,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                handles=reversed(handles),
                labels=reversed(labels),
            )
        if istream == 1:
            ax.set_ylabel("")
            ax.get_legend().remove()

        ax.set_title(stream)

        fig.suptitle(
            (
                "large dots without outline = data (error = SEM)\n"
                "small dots with black outline = model (error not plotted)"
                f" {slug}"
            )
        )

sns.despine(fig)
fig.tight_layout()

# %%
# plot weights over positions: positions as hue
data = posweightdata[posweightdata["position"].isin([0, 9])]
g = sns.catplot(
    x="number",
    y="weight",
    hue="position",
    data=data,
    col="stream",
    row="weight_type",
    kind="point",
    dodge=False,
    ci=68,
    palette="crest_r",
)

g.fig.suptitle(f"Model based on initial guesses of type '{x0_type}' {slug}", y=1.05)
fname = analysis_dir / "figures" / f"posweights_positionhue{slug}.jpg"
g.fig.savefig(fname)

# %%
# save weights to files
sortby = ["subject", "stream", "weight_type", "number"]
weightdata = weightdata.sort_values(by=sortby).reset_index(drop=True)
posweightdata = posweightdata.sort_values(
    by=sortby[:-1] + ["position", "number"]
).reset_index(drop=True)


fname = analysis_dir / "derived_data" / f"weights{slug}.tsv"
weightdata.to_csv(fname, sep="\t", na_rep="n/a", index=False)

fname = analysis_dir / "derived_data" / f"posweights{slug}.tsv"
posweightdata.to_csv(fname, sep="\t", na_rep="n/a", index=False)

# %%
