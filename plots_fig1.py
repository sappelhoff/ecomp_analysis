"""Figure 1 plot - paradigm figure and behavior."""

# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import ANALYSIS_DIR_LOCAL, STREAMS
from utils import eq1, find_dot_idxs

# %%
# Settings
analysis_dir = ANALYSIS_DIR_LOCAL


ci = 68

subj_line_settings = dict(color="black", alpha=0.1, linewidth=0.75)
axhline_args = dict(color="black", linestyle="--", linewidth=1)
pointscale = 1
pointmarkers = "."  # for pointplot which marker style
pointerrwidth = 3
pointlinewidth = axhline_args["linewidth"]
pointcapwidth = 0.1
swarmsize = 2

minimize_method = "Nelder-Mead"
x0_type = "specific"
# %%
# File paths
fname_accs = analysis_dir / "derived_data" / "accuracies.tsv"

fname_weights = analysis_dir / "derived_data" / "weights.tsv"
fname_weights_k_is_1 = analysis_dir / "derived_data" / "weights_k_is_1.tsv"

fname_estimates = analysis_dir / "derived_data" / f"estim_params_{minimize_method}.tsv"

fname_fig1 = analysis_dir / "figures" / "fig1b+.pdf"

# %%
# Load param estimates
df_estims = pd.read_csv(fname_estimates, sep="\t")
df_estims = df_estims[df_estims["x0_type"] == x0_type]

# %%
# Figure 1b+
# Figure 1a is created in LibreOffice Draw
# Fig1a and Fig1b+ are then stitched together in Latex

# figure layout
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.tight_layout(h_pad=2.25)
# %%
# panel b - accuracies
df_accs = pd.read_csv(fname_accs, sep="\t")
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
    _idx1, _idx2 = find_dot_idxs(ax, int(df_accs.shape[0] / 2))
    locs1 = ax.get_children()[_idx1].get_offsets()
    locs2 = ax.get_children()[_idx2].get_offsets()

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
df_ws = pd.read_csv(fname_weights, sep="\t")
df_ws = df_ws[df_ws["weight_type"].isin(["data", "model", "model_k1"])]

df_ws_k1 = pd.read_csv(fname_weights_k_is_1, sep="\t")
df_ws_k1 = df_ws_k1[df_ws_k1["weight_type"].isin(["data", "model", "model_k1"])]

with sns.plotting_context("talk"):

    for istream, stream in enumerate(STREAMS):
        data = df_ws[df_ws["stream"] == stream]

        x = "number"
        colname = "weight"
        ax = axs[0, 1 + istream]

        sns.pointplot(
            data=data[data["weight_type"] == "data"],
            x=x,
            y=colname,
            ax=ax,
            ci=68,
            color=f"C{istream}",
        )

        sns.pointplot(
            data=data[data["weight_type"] == "model"],
            x=x,
            y=colname,
            ax=ax,
            ci=None,
            color="black",
            join=False,
            scale=0.5,
        )
        plt.setp(ax.collections, zorder=100, label="")

        plot_k1_fitted = True  # whether to plot "fitted" k1, or by just setting k=1
        if plot_k1_fitted:
            ____ = df_ws_k1[df_ws_k1["stream"] == stream]
            k1_true = ax.plot(
                np.arange(9),
                ____[____["weight_type"] == "model_k1"]
                .groupby("number")["weight"]
                .mean(),
                color="black",
                lw=0.5,
            )
        else:
            k1 = ax.plot(
                np.arange(9),
                data[data["weight_type"] == "model_k1"]
                .groupby("number")["weight"]
                .mean(),
                color="black",
                lw=0.5,
            )

        ax.axhline(0.5, xmax=0.95, **axhline_args)

        # plot insets
        # bias = df_estims[df_estims["stream"] == stream]["bias"].mean()
        kappa = df_estims[df_estims["stream"] == stream]["kappa"].mean()
        xs = np.linspace(-1, 1, 9)  # "numbers_rescaled"
        ys = eq1(X=xs, bias=0, kappa=kappa)
        size = 1.0
        axins = inset_axes(
            ax,
            width=size,
            height=size,
            loc="upper left",
            bbox_to_anchor=(0.02, 1),
            bbox_transform=ax.transAxes,
        )
        axins.plot(xs, ys, color=f"C{STREAMS.index(stream)}")

        # tmp: plot other curve in same plot
        _kappa = df_estims[
            df_estims["stream"]
            == f"{STREAMS[dict(zip((0, 1), (1, 0)))[STREAMS.index(stream)]]}"
        ]["kappa"].mean()
        _ys = eq1(X=xs, bias=0, kappa=_kappa)
        axins.plot(
            xs,
            _ys,
            color=f"C{dict(zip((0, 1), (1, 0)))[STREAMS.index(stream)]}",
            zorder=-1,
            linestyle="--",
            linewidth=0.75,
        )
        # tmp over

        # tmp 2: draw lines through 1,2 and 8,9 and 4,6 --> y=mx+c
        y1 = data[(data["weight_type"] == "data") & (data["number"] == 1)][
            "weight"
        ].mean()
        y2 = data[(data["weight_type"] == "data") & (data["number"] == 2)][
            "weight"
        ].mean()
        y4 = data[(data["weight_type"] == "data") & (data["number"] == 4)][
            "weight"
        ].mean()
        y6 = data[(data["weight_type"] == "data") & (data["number"] == 6)][
            "weight"
        ].mean()
        y8 = data[(data["weight_type"] == "data") & (data["number"] == 8)][
            "weight"
        ].mean()
        y9 = data[(data["weight_type"] == "data") & (data["number"] == 9)][
            "weight"
        ].mean()

        _kwargs = dict(color="red", lw=0.75, zorder=100)

        _coefs = np.polyfit((0, 1), (y1, y2), 1)
        polynomial = np.poly1d(_coefs)
        ax.plot(np.linspace(-3, 4), polynomial(np.linspace(-3, 4)), **_kwargs)

        _coefs = np.polyfit((3, 5), (y4, y6), 1)
        polynomial = np.poly1d(_coefs)
        ax.plot(np.linspace(0, 8), polynomial(np.linspace(0, 8)), **_kwargs)

        _coefs = np.polyfit((7, 8), (y8, y9), 1)
        polynomial = np.poly1d(_coefs)
        ax.plot(np.linspace(4, 11), polynomial(np.linspace(4, 11)), **_kwargs)
        # tmp over

        axins.axhline(0, lw=0.1, color="black")
        sns.despine(ax=axins)
        axins.set_xticks([])
        axins.set_yticks([])

        # inset label
        axins.text(
            x=0.175,
            y=0.675,
            s=r"$\widehat{k}$" + f"$ = {kappa:.2f}$",
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            zorder=100,
        )

        # make a legend
        if istream == 0:
            handles = [
                Line2D([0], [0], color=f"C{istream}", marker="o", lw=4),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="white",
                    markerfacecolor="black",
                    markersize=8,
                ),
                Line2D([0], [0], color="black"),
            ]

            ax.legend(
                handles=handles,
                labels=["Data", "Model", "Model (k=1)"],
                loc="lower right",
                ncol=1,
                frameon=False,
                fontsize=12,
            )

        # other settings
        sns.despine(ax=ax)
        ax.set(xlabel="", ylabel="Decision weight")
        ax.set_title(stream.capitalize())

for ax in axs[0, 1:]:
    ax.set_ylim(
        (
            min(axs[0, 1].get_ylim()[0], axs[0, 2].get_ylim()[0]),
            max(axs[0, 1].get_ylim()[1], axs[0, 2].get_ylim()[1]),
        )
    )


# %%
# panels e, f, g - kappa, bias, noise
param_names = ["kappa", "bias", "noise"]
df_estims = df_estims[["subject", "stream"] + param_names].melt(
    id_vars=["subject", "stream"], var_name="parameter"
)

order = STREAMS
x = "stream"
colname = "value"
hlines = dict(kappa=1, bias=0, noise=0)
with sns.plotting_context("talk"):

    for iparam, param in enumerate(param_names):
        ax = axs[1, iparam]

        data = df_estims[df_estims["parameter"] == param]

        sns.pointplot(
            data=data,
            x=x,
            order=order,
            y=colname,
            ci=68,
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

        # connect subj dots with lines
        # https://stackoverflow.com/a/51157346/5201771
        _idx1, _idx2 = find_dot_idxs(ax, int(df_accs.shape[0] / 2))
        locs1 = ax.get_children()[_idx1].get_offsets()
        locs2 = ax.get_children()[_idx2].get_offsets()

        # before plotting, we need to sort so that the data points correspond
        sort_idxs1 = np.argsort(data[data["stream"] == STREAMS[0]][colname].to_numpy())
        sort_idxs2 = np.argsort(data[data["stream"] == STREAMS[1]][colname].to_numpy())
        locs2_sorted = locs2[sort_idxs2.argsort()][sort_idxs1]

        for i in range(locs1.shape[0]):
            _x = [locs1[i, 0], locs2_sorted[i, 0]]
            _y = [locs1[i, 1], locs2_sorted[i, 1]]
            ax.plot(_x, _y, **subj_line_settings)

        ax.axhline(hlines[param], xmax=0.95, **axhline_args)
        sns.despine(ax=ax)

        ax.text(
            x=0.5,
            y=0.9,
            s=param,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ylabel = dict(kappa="$k$", bias="$b$", noise="$s$")[param]
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels([i.capitalize() for i in STREAMS])

# %%
# Final settings and save
# axs[1, 0].set_yscale("log")
# axs[1, 0].set_ylim(0.2, None)
# Save the figure
fig.savefig(fname_fig1, bbox_inches="tight")

# %%
