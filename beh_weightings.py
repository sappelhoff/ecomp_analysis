"""Analyze weights."""
# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    ANALYSIS_DIR_LOCAL,
    CHOICE_MAP,
    DATA_DIR_LOCAL,
    NUMBERS,
    STREAMS,
    SUBJS,
)
from utils import (
    eq1,
    get_estim_params,
    get_sourcedata,
    prep_model_inputs,
    prep_weight_calc,
    psychometric_model,
)

# %%
# Settings
numbers_rescaled = np.interp(NUMBERS, (NUMBERS.min(), NUMBERS.max()), (-1, +1))

positions = np.arange(10)

minimize_method = "Nelder-Mead"

# %%
# Prepare file paths
analysis_dir = ANALYSIS_DIR_LOCAL
data_dir = DATA_DIR_LOCAL

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


def calc_CP_weights(sub, stream, x0_type, data_dir, analysis_dir, minimize_method):
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
    parameters = get_estim_params(sub, stream, x0_type, minimize_method, analysis_dir)

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

# Take model parameter estimates based on "fixed" or "specific" initial guesses
x0_type = "specific"

wtypes = ["data", "model"]
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
                    x0_type,
                    data_dir,
                    analysis_dir,
                    minimize_method,
                )
            else:
                raise RuntimeError("unrecognized `wtype`")

            # save in DF
            wdf = pd.DataFrame.from_dict(
                dict(
                    subject=sub,
                    stream=stream,
                    number=NUMBERS,
                    weight=weights,
                    weight_type=wtype,
                )
            )
            pwdf = pd.DataFrame.from_dict(
                dict(
                    subject=sub,
                    stream=stream,
                    number=np.tile(NUMBERS, len(positions)),
                    position=np.repeat(positions, len(NUMBERS)),
                    weight=position_weights.flatten(),  # shape(pos X weight)
                    weight_type=wtype,
                )
            )

            weight_dfs.append(wdf)
            posweight_dfs.append(pwdf)

weightdata = pd.concat(weight_dfs).reset_index(drop=True)
posweightdata = pd.concat(posweight_dfs).reset_index(drop=True)
# %%
# save to files
# (save only nonp weights for now)


fname = analysis_dir / "derived_data" / "weights.tsv"
weightdata[weightdata["weight_type"] == "data"].to_csv(
    fname, columns=weightdata.columns[:-1], sep="\t", na_rep="n/a", index=False
)

fname = analysis_dir / "derived_data" / "posweights.tsv"
posweightdata[posweightdata["weight_type"] == "data"].to_csv(
    fname, columns=posweightdata.columns[:-1], sep="\t", na_rep="n/a", index=False
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
plot_ref_lines = False
if plot_ref_lines and not plotgrid:
    ax.axhline(0.5, linestyle="--", color="black", lw=0.5)
    ax.axvline(4, linestyle="--", color="black", lw=0.5)
    ax.plot(np.arange(9), np.linspace(0, 1, 9), linestyle="--", color="black", lw=0.5)
    fname = str(fname).replace(".jpg", "_reflines.jpg")

fname = analysis_dir / "figures" / "weights.jpg"
if plotgrid:
    sns.despine(g.fig)
    g.fig.suptitle(f"Model based on initial guesses of type '{x0_type}'", y=1.05)
    g.fig.savefig(fname)
else:
    sns.despine(fig)
    fig.savefig(fname)

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

g.fig.suptitle(f"Model based on initial guesses of type '{x0_type}'", y=1.05)
fname = analysis_dir / "figures" / "posweights_numberhue.jpg"
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
            )
        )

sns.despine(fig)
fig.tight_layout()
# %%


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

g.fig.suptitle(f"Model based on initial guesses of type '{x0_type}'", y=1.05)
fname = analysis_dir / "figures" / "posweights_positionhue.jpg"
g.fig.savefig(fname)

# %%
# Plotting over- and underweighting according to model
# numberline
cmap = sns.color_palette("crest_r", as_cmap=True)

bs = np.linspace(-1, 1, 7)
ks = np.linspace(0.4, 3, 5)

fig, axs = plt.subplots(1, len(bs), figsize=(10, 5), sharex=True, sharey=True)
for i, b in enumerate(bs):
    ax = axs.flat[i]
    ax.plot(
        np.linspace(-1, 1, 9),
        eq1(numbers_rescaled, bias=0, kappa=1),
        color="k",
        marker="o",
        label="b=0, k=1",
    )
    ax.set(title=f"bias={b:.2}", ylabel="decision weight", xlabel="X")

    for k in ks:
        ax.plot(
            np.linspace(-1, 1, 1000),
            eq1(np.linspace(-1, 1, 1000), b, k),
            color=cmap(k / ks.max()),
            label=f"k={k:.2f}",
        )

    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(0, color="k", linestyle="--")
    if i == 0:
        ax.legend()

fig.tight_layout()

# %%
