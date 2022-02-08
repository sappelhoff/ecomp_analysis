"""Analyze accuracies.

TODO: Ignore "ambiguous" trials for accuracy.
"""
# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
import seaborn as sns
from scipy.stats import sem

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_LOCAL
from utils import get_sourcedata

# %%
# Settings

analysis_dir = ANALYSIS_DIR_LOCAL

# %%
# Get accuracies from participants
data = dict(sub=[], stream=[], acc=[])
for sub in range(1, 33):

    # skip bad subjs
    if sub in BAD_SUBJS:
        continue

    for stream in ["single", "dual"]:
        _, tsv = get_sourcedata(sub, stream, DATA_DIR_LOCAL)
        df = pd.read_csv(tsv, sep="\t")

        corrects = df["correct"].to_numpy()

        # drop invalid trials (timeouts)
        dropidxs = np.nonzero(~df["validity"].to_numpy())[0]
        corrects = np.delete(corrects, dropidxs)

        data["sub"] += [sub]
        data["stream"] += [stream]
        data["acc"] += [np.mean(corrects)]

df_acc = pd.DataFrame.from_dict(data)
fname = analysis_dir / "derived_data" / "accuracies.tsv"
df_acc.to_csv(fname, sep="\t", na_rep="n/a", index=False)

# %%

# settings
plotting_context = dict(context="talk", font_scale=1)

swarmsize = 2  # for strip and swarmplot how big the dots

ci = 68  # the error bars, 68 ~ SEM

axhline_args = dict(color="black", linestyle="--", linewidth=1)

pointscale = 3
pointmarkers = "."  # for pointplot which marker style
pointerrwidth = 3
pointlinewidth = axhline_args["linewidth"]
pointcapwidth = 0.1

labelpad = 12

stream_order = ["single", "dual"]

subj_line_settings = dict(color="black", alpha=0.1, linewidth=0.75)

panel_letter_kwargs = dict(
    x=-0.2,
    y=1.15,
    horizontalalignment="center",
    verticalalignment="center",
)


with sns.plotting_context(**plotting_context):

    fig, ax = plt.subplots(figsize=(5, 5))

    x = "stream"
    order = stream_order
    colname = "acc"
    ax = ax
    data = df_acc

    with plt.rc_context({"lines.linewidth": pointlinewidth}):
        sns.pointplot(
            x=x,
            order=order,
            y=colname,
            data=data,
            ci=ci,
            ax=ax,
            markers=pointmarkers,
            scale=pointscale,
            errwidth=pointerrwidth,
            capsize=pointcapwidth,
            color="black",
        )

    sns.swarmplot(
        x=x,
        order=order,
        y=colname,
        data=data,
        ax=ax,
        size=swarmsize,
    )

    # ax.set_ylim(0.5, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(-0.75, 1.75)

    # connect subj dots with lines
    # https://stackoverflow.com/a/51157346/5201771
    idx0 = 1
    idx1 = 2
    locs1 = ax.get_children()[idx0].get_offsets()
    locs2 = ax.get_children()[idx1].get_offsets()

    # before plotting, we need to sort so that the data points correspond
    sort_idxs1 = np.argsort(data[data["stream"] == stream_order[0]]["acc"].to_numpy())
    sort_idxs2 = np.argsort(data[data["stream"] == stream_order[1]]["acc"].to_numpy())
    locs2_sorted = locs2[sort_idxs2.argsort()][sort_idxs1]

    for i in range(locs1.shape[0]):
        _x = [locs1[i, 0], locs2_sorted[i, 0]]
        _y = [locs1[i, 1], locs2_sorted[i, 1]]
        ax.plot(_x, _y, **subj_line_settings)

    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xticklabels([i.capitalize() for i in stream_order])

    sns.despine(fig)
    fname = analysis_dir / "figures" / "accs.jpg"
    fig.savefig(fname)


# %%
# print descriptives
descr = df_acc.groupby("stream")["acc"].agg([np.mean, sem]).round(3)
descr

# %%
# print stats
tstats = pingouin.ttest(
    df_acc[df_acc["stream"] == "single"]["acc"],
    df_acc[df_acc["stream"] == "dual"]["acc"],
    paired=True,
)
tstats

# %%
# print acc range
df_acc["acc"].agg(["min", "max"]).round(3).to_list()

# %%
# Print earned bonus money
# see:
# https://github.com/sappelhoff/ecomp_experiment/blob/main/ecomp_experiment/utils.py#L146
# < 55 = 0€, >= 90€ = 10€, all in between mapped linearly between 0€ and 10€
accs = np.ceil(df_acc.groupby("sub").mean().to_numpy() * 100).astype(int)
assert not (accs < 55).any()
assert not (accs >= 90).any()
cents_map = np.linspace(0, 1000, 90 - 55)
bonus_cents = cents_map[accs - 55]
bonus_euro = np.ceil(bonus_cents / 100).astype(int).flatten()
print(f"Participant earned €{bonus_euro.mean():.2f}±{bonus_euro.std():.2f} on average.")

# %%
