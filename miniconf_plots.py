"""Plots for the miniconference 2021_2 at ARC."""

# %% Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ANALYSIS_DIR_LOCAL, NUMBERS

# %%
# Settings

analysis_dir = ANALYSIS_DIR_LOCAL
# %%

# load data
fname = analysis_dir / "derived_data" / "weights.tsv"
weightdata = pd.read_csv(fname, sep="\t")

fname = analysis_dir / "derived_data" / "accuracies.tsv"
accdata = pd.read_csv(fname, sep="\t")

# calc model weights
numbers_rescaled = np.interp(NUMBERS, (NUMBERS.min(), NUMBERS.max()), (-1, +1))


def eq1_rescale(X, bias, kappa):
    """See equation 1 from Spitzer et al. 2017, Nature Human Behavior."""
    dv = np.sign(X + bias) * (np.abs(X + bias) ** kappa)

    dv = np.interp(dv, (dv.min(), dv.max()), (0, 1))
    return dv


# %%
# plot settings
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

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    (ax1, ax2, ax3) = axs

    # Accuracies
    x = "stream"
    order = stream_order
    colname = "acc"
    ax = ax1
    data = accdata

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

    # model vals
    ax2.plot(
        np.linspace(1, 9, 9),
        eq1_rescale(numbers_rescaled, bias=0, kappa=0.5),
        color="C0",
        label="compressed",
    )
    ax2.plot(
        np.linspace(1, 9, 9),
        eq1_rescale(numbers_rescaled, bias=0, kappa=2),
        color="C1",
        label="anti-compressed",
    )
    ax2.plot(
        np.linspace(1, 9, 9),
        eq1_rescale(numbers_rescaled, bias=0, kappa=1),
        color="black",
        label="linear",
    )
    ax2.legend(frameon=False)
    ax2.set(ylabel="Decision Weight", xlabel="Number")
    ax2.axhline(0.5, linestyle="--", color="black", lw=0.5)
    ax2.locator_params(nbins=9, axis="x")

    # weightings
    sns.pointplot(
        x="number",
        y="weight",
        hue="stream",
        data=weightdata,
        ax=ax3,
        dodge=False,
        ci=68,
    )
    ax3.axhline(0.5, linestyle="--", color="black", lw=0.5)
    ylim_absmax = np.abs(ax3.get_ylim()).max() - 0.5
    ax3.set(
        xlabel="Number",
        ylabel="Decision Weight",
        ylim=(0.5 - ylim_absmax, 0.5 + ylim_absmax),
    )
    ax3.legend(title="Stream")
    ax3.get_legend().get_texts()[0].set_text("Single")
    ax3.get_legend().get_texts()[1].set_text("Dual")

    sns.despine(fig)
    fig.tight_layout()

    # plot panel letters
    # see: https://gitter.im/matplotlib/matplotlib?at=5fb53257c6fe0131d40227bb
    extra_artists = []
    for ax, label in zip(axs.flat, (i for i in "abc")):

        x = -0.5
        y = 0.3
        offset = matplotlib.transforms.ScaledTranslation(x, y, fig.dpi_scale_trans)
        transform = ax.transAxes + offset

        ax.text(
            s=label,
            x=0,
            y=1,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=plt.rcParams["font.size"] + 2,
            transform=transform,
        )
        extra_artists.append(ax)

fname = analysis_dir / "figures" / "miniconf.jpg"
fig.savefig(fname, bbox_inches="tight", bbox_extra_artists=extra_artists)
# %%

# compute RDMs
rdms = []
for kappa in [1, 0.5, 2]:
    xx = eq1_rescale(numbers_rescaled, bias=0, kappa=kappa)
    arrs = list()
    for num in xx:
        arrs.append(np.abs(xx - num))
    rdm = np.stack(arrs, axis=0)

    rdm = rdm / rdm.max()
    assert np.isclose(rdm.min(), 0)
    assert np.isclose(rdm.max(), 1)

    rdms.append(rdm)

# plot
ticklabels = [str(i) for i in range(11)]

with sns.plotting_context(**plotting_context):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for ax, rdm, title in zip(axs, rdms, ["linear", "compressed", "anti-compressed"]):
        im = ax.imshow(rdm)
        ax.set_title(title)

    ax.locator_params(nbins=9)
    _ = ax.set(xticklabels=ticklabels, yticklabels=ticklabels)

    fig.tight_layout()

    cbar = fig.colorbar(
        im, ax=axs.ravel().tolist(), shrink=0.75, label="distance", pad=0.025
    )
    cbar.set_ticks(np.arange(0, 1.1, 0.5))
    cbar.set_ticklabels(["low", "medium", "high"])

extra_artists = [cbar]
fname = analysis_dir / "figures" / "miniconf2.jpg"
fig.savefig(fname, bbox_extra_artists=extra_artists)

# %%
