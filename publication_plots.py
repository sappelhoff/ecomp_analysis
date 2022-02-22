"""Publication plots."""

# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ANALYSIS_DIR_LOCAL, STREAMS

# %%
# Settings
analysis_dir = ANALYSIS_DIR_LOCAL


ci = 68

subj_line_settings = dict(color="black", alpha=0.1, linewidth=0.75)
axhline_args = dict(color="black", linestyle="--", linewidth=1)
pointscale = 3
pointmarkers = "."  # for pointplot which marker style
pointerrwidth = 3
pointlinewidth = axhline_args["linewidth"]
pointcapwidth = 0.1
swarmsize = 2
# %%
# File paths
fname_accs = analysis_dir / "derived_data" / "accuracies.tsv"

fname_fig1 = analysis_dir / "figures" / "fig1b+.pdf"
# %%
# Figure 1b+
# Figure 1a is created in LibreOffice Draw
# Fig1a and Fig1b+ are then stitched together in Latex

# figure layout
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# %%
# panel b - accuracies
df_accs = pd.read_csv(fname_accs, sep="\t")
df_accs
with sns.plotting_context("talk"):

    x = "stream"
    order = STREAMS
    colname = "accuracy"
    ax = axs[0, 0]
    data = df_accs

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
    sort_idxs1 = np.argsort(data[data["stream"] == STREAMS[0]]["accuracy"].to_numpy())
    sort_idxs2 = np.argsort(data[data["stream"] == STREAMS[1]]["accuracy"].to_numpy())
    locs2_sorted = locs2[sort_idxs2.argsort()][sort_idxs1]

    for i in range(locs1.shape[0]):
        _x = [locs1[i, 0], locs2_sorted[i, 0]]
        _y = [locs1[i, 1], locs2_sorted[i, 1]]
        ax.plot(_x, _y, **subj_line_settings)

    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xticklabels([i.capitalize() for i in STREAMS])

    sns.despine(ax=ax)

# %%
# panels c and d - weightings


# %%
# panels e, f, g - kappa, bias, noise


# %%
# Save the figure
fig.savefig(fname_fig1)

# %%
