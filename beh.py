"""Beh analysis."""
# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ANALYSIS_DIR, BAD_SUBJS, DATA_DIR_LOCAL, get_sourcedata

# %%
# Check how often participants "timed out" on their choices
for sub in range(1, 33):
    for stream in ["single", "dual"]:
        _, tsv = get_sourcedata(sub, stream, DATA_DIR_LOCAL)
        df = pd.read_csv(tsv, sep="\t")
        validity_sum = df["validity"].to_numpy().sum()
        if validity_sum != df.shape[0]:
            print(sub, stream, f" - n timeouts: {df.shape[0]-validity_sum}")

# %%


def calc_nonp_weights(df):
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

    Returns
    -------
    weights : np.ndarray, shape(9,)
        The weight for each of the 9 numbers in ascending
        order (1 to 9).
    position_weights : np.ndarray, shape(10, 9)
        The weight for each of the 9 numbers in ascending
        order, calculated for each of the 10 sample positions.
    """
    # work on a copy of the data
    weights_df = df.copy()

    stream = np.unique(weights_df["stream"])[0]

    # remove NaN rows
    nan_row_idxs = np.nonzero(~weights_df["validity"].to_numpy())[0]
    for idx in nan_row_idxs:
        assert pd.isna(weights_df.loc[idx, "direction"])
    weights_df = weights_df.drop(nan_row_idxs)

    # prepare data
    nsamples = 10
    isamples = [f"sample{i}" for i in range(1, nsamples + 1)]  # samples 1 to nsamples
    samples_signed = weights_df.loc[:, isamples].to_numpy().reshape(-1)
    samples = np.abs(samples_signed)
    colors = (np.sign(samples_signed) + 1) / 2  # color1=0, color2=1
    positions = np.tile(
        np.arange(len(isamples), dtype=int), len(weights_df["trial"])
    )  # sample positions
    assert positions.shape == samples.shape

    if stream == "single":
        # sanity check
        np.testing.assert_array_equal(
            np.unique(weights_df["choice"]), ["higher", "lower"]
        )

        # map lower/higher to 0/1 ... and repeat choice for each sample
        choices = np.repeat(
            weights_df["choice"].map({"lower": 0, "higher": 1}).to_numpy(),
            len(isamples),
        )
        assert choices.shape == samples.shape
    else:
        assert stream == "dual"
        # sanity check
        np.testing.assert_array_equal(np.unique(weights_df["choice"]), ["blue", "red"])

        # map red/blue to 0/1 ... and repeat choice for each sample
        choices = np.repeat(
            weights_df["choice"].map({"red": 0, "blue": 1}).to_numpy(), len(isamples)
        )
        assert choices.shape == samples.shape

    # Calculate overall weights and for each sample position
    numbers = np.arange(1, 10, dtype=int)  # numbers 1 to 9 were shown
    weights = np.zeros(len(numbers))
    position_weights = np.zeros((len(isamples), len(numbers)))
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


# %%

# calculate weights over subjects
numbers = np.arange(1, 10, dtype=int)
positions = np.arange(10)
weight_dfs = []
posweight_dfs = []
for sub in range(1, 33):

    # skip bad subjs
    if sub in BAD_SUBJS:
        continue

    for stream in ["single", "dual"]:

        _, tsv = get_sourcedata(sub, stream, DATA_DIR_LOCAL)
        df = pd.read_csv(tsv, sep="\t")

        weights, position_weights = calc_nonp_weights(df)

        # save in DF
        wdf = pd.DataFrame.from_dict(
            dict(sub=sub, stream=stream, number=numbers, weight=weights)
        )
        pwdf = pd.DataFrame.from_dict(
            dict(
                sub=[sub] * len(positions) * len(numbers),
                stream=[stream] * len(positions) * len(numbers),
                number=np.tile(numbers, len(positions)),
                position=np.tile(positions, len(numbers)),
                weight=position_weights.reshape(-1),
            )
        )

        weight_dfs.append(wdf)
        posweight_dfs.append(pwdf)

weightdata = pd.concat(weight_dfs)
posweightdata = pd.concat(posweight_dfs)

# plot weights
fig, ax = plt.subplots()
sns.pointplot(
    x="number", y="weight", hue="stream", data=weightdata, ax=ax, dodge=False, ci=68
)
ax.axhline(0.5, linestyle="--", color="black", lw=0.5)

fname = ANALYSIS_DIR / "figures" / "weights.jpg"


# plot regression lines per task
_tmp = weightdata.groupby(["stream", "number"])["weight"].mean().reset_index()
for _stream, _color in zip(["single", "dual"], ["C0", "C1"]):
    xy = _tmp[_tmp["stream"] == _stream][["number", "weight"]].to_numpy()
    m, b = np.polyfit(xy[:, 0], xy[:, 1], 1)
    plt.plot(np.arange(9), m * xy[:, 0] + b, color=_color)

# optional horizontal, vertical, and diagonal (=linear weights) reference lines
plot_ref_lines = False
if plot_ref_lines:
    ax.axhline(0.5, linestyle="--", color="black", lw=0.5)
    ax.axvline(4, linestyle="--", color="black", lw=0.5)
    ax.plot(np.arange(9), np.linspace(0, 1, 9), linestyle="--", color="black", lw=0.5)
    fname = str(fname).replace(".jpg", "_reflines.jpg")

sns.despine(fig)
fig.savefig(fname)


# %%
# plot weights over positions: numbers as hue
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for stream, ax in zip(["single", "dual"], axs):

    sns.pointplot(
        x="position",
        y="weight",
        hue="number",
        data=posweightdata[posweightdata["stream"] == stream],
        ax=ax,
        dodge=False,
        ci=68,
        palette="crest_r",
    )
    ax.set_title(stream)
    ax.set_ylim(None, 0.85)
    ax.axhline(0.5, linestyle="--", color="black", lw=0.5)
    if stream == "single":
        ax.get_legend().remove()
    else:
        ax.legend(ncol=5, title="number")

fig.tight_layout()
sns.despine(fig)
fname = ANALYSIS_DIR / "figures" / "posweights_numberhue.jpg"
fig.savefig(fname)

# %%
# plot weights over positions: positions as hue
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for stream, ax in zip(["single", "dual"], axs):

    data = posweightdata[
        (posweightdata["stream"] == stream) & (posweightdata["position"].isin([0, 9]))
    ]
    # data = posweightdata[posweightdata["stream"] == stream]  # this is too crowded

    sns.pointplot(
        x="number",
        y="weight",
        hue="position",
        data=data,
        ax=ax,
        dodge=False,
        ci=68,
        palette="crest_r",
    )
    ax.set_title(stream)
    ax.axhline(0.5, linestyle="--", color="black", lw=0.5)
    if stream == "single":
        ax.get_legend().remove()
    else:
        ax.legend(ncol=5, title="position")

fig.tight_layout()
sns.despine(fig)
fname = ANALYSIS_DIR / "figures" / "posweights_positionhue.jpg"
fig.savefig(fname)
# %%
