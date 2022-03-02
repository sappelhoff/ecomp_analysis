"""Figure 4 plot - CPP/P3."""
# %%
# Imports
import json
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import ANALYSIS_DIR_LOCAL, NUMBERS, STREAMS
from utils import eq1

# %%
# Settings
analysis_dir = ANALYSIS_DIR_LOCAL

axhline_args = dict(color="black", linestyle="--", linewidth=1)

numbers_rescaled = np.interp(NUMBERS, (NUMBERS.min(), NUMBERS.max()), (-1, +1))

# %%
# Prepare file paths
fname_fig4 = analysis_dir / "figures" / "fig4_pre.pdf"

fname_erps = analysis_dir / "derived_data" / "erps.tsv"
fname_amps = analysis_dir / "derived_data" / "erp_amps.tsv"
fname_adm = analysis_dir / "derived_data" / "erp_adm.tsv"

fname_permres_erps = analysis_dir / "derived_data" / "erp_perm_results.json"

# %%
# Load data
df_erps = pd.read_csv(fname_erps, sep="\t")
df_amps = pd.read_csv(fname_amps, sep="\t")
df_adm = pd.read_csv(fname_adm, sep="\t")

baseline = eval(df_amps["baseline"][0])
mean_times = eval(df_amps["mean_times"][0])
p3_group = eval(df_amps["p3_group"][0])

with open(fname_permres_erps, "r") as fin:
    permdistr_dict = json.load(fin)

# prepare MNE montage
montage = mne.channels.make_standard_montage("easycap-M1")
info = mne.create_info(p3_group, 1, ch_types="eeg")
info = info.set_montage(montage)

# %%
# Start figure
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.tight_layout(h_pad=4, w_pad=3)

# %%
# plot ERPs
times = np.unique(df_erps["time"])
palette = "crest_r"
colors = list(sns.color_palette(palette, n_colors=len(NUMBERS)))
with sns.plotting_context("talk"):
    for istream, stream in enumerate(STREAMS):
        ax = axs[0, istream]
        data = df_erps[df_erps["stream"] == stream]
        sns.lineplot(
            x="time",
            y="value",
            hue="number",
            data=data,
            ci=68,
            ax=ax,
            palette=palette,
        )

        sns.despine(ax=ax)
        ylabel = "Amplitude (µV)" if istream == 0 else ""
        ax.set(xlabel="Time (s)", ylabel=ylabel)
        ax.axhline(0, **axhline_args)
        ax.axvline(0, **axhline_args)
        ax.axvspan(*mean_times, color="black", alpha=0.1)

        if istream == 1:
            handles = []
            for inum, num in enumerate(NUMBERS[::-1]):
                handles.append(
                    Line2D([0], [0], color=colors[::-1][inum], label=str(num))
                )
            ax.legend(
                handles=handles,
                loc="upper left",
                title=None,
                frameon=False,
                bbox_to_anchor=(1, 1),
            )
        else:
            ax.get_legend().remove()

        # plot insets
        size = 1.25
        axins = inset_axes(
            ax,
            width=size,
            height=size,
            loc="upper left",
            bbox_to_anchor=(0.12, 1),
            bbox_transform=ax.transAxes,
        )
        mne.viz.plot_sensors(info, kind="topomap", title="", axes=axins, show=False)

        ax.text(
            x=0.5,
            y=0.9,
            s=stream.capitalize(),
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    for ax in axs[0, ...]:
        ax.set_ylim(
            (
                min(axs[0, 0].get_ylim()[0], 1.1 * axs[0, 1].get_ylim()[0]),
                max(axs[0, 0].get_ylim()[1], 1.1 * axs[0, 1].get_ylim()[1]),
            )
        )

    # plot significance bars
    for istream, stream in enumerate(STREAMS):
        ax = axs[0, istream]
        y = ax.get_ylim()[0] * 0.9
        clusters = permdistr_dict[stream]["sig_clusters"]
        for clu in clusters:
            ax.plot(times[clu], [y] * len(clu), c="black", ls="-")

# %%
# plot mean amplitudes
use_smalltitle = False
with sns.plotting_context("talk"):
    ax = axs[1, 0]
    sns.pointplot(
        x="number", y="mean_amp", hue="stream", data=df_amps, ci=68, ax=ax, dodge=False
    )
    sns.despine(ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title=None,
        handles=handles,
        labels=[i.capitalize() for i in labels],
        frameon=False,
    )
    ax.set(ylabel="Amplitude (µV)", xlabel="Sample")

    if use_smalltitle:
        ax.text(
            x=0.5,
            y=0.9,
            s=f"CPP/P3 amplitudes\n{mean_times[0]} - {mean_times[1]} s",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        title = "CPP/P3 amplitudes"
        ax.set_title(title, fontweight="bold")
        ax.text(
            x=0.5,
            y=0.9,
            s=f"{mean_times[0]} - {mean_times[1]} s",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


# %%
# Plot ADM
draw_subjs_d = False
with sns.plotting_context("talk"):
    ax = axs[1, 1]
    data = df_adm
    if draw_subjs_d:
        sns.pointplot(
            x="number", y="dv_abs", hue="stream", data=data, ci=68, ax=ax, dodge=False
        )
    else:
        for istream, stream in enumerate(STREAMS):
            b = df_adm[df_adm["stream"] == stream]["mapmax_bias"].to_numpy()[0]
            k = df_adm[df_adm["stream"] == stream]["mapmax_kappa"].to_numpy()[0]
            vals = np.abs(eq1(numbers_rescaled, bias=b, kappa=k))
            ax.plot(NUMBERS - 1, vals, marker="o")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="FixedFormatter .* FixedLocator"
            )
            ax.set_xticklabels([""] + NUMBERS.tolist())

    draw_inset_d = True
    if draw_inset_d:
        size = 1.75
        anchor = (0.08, 1)
        if not use_smalltitle:
            anchor = (0.25, 1)
            size = 2
        axins = inset_axes(
            ax,
            width=size,
            height=size,
            loc="upper left",
            bbox_to_anchor=anchor,
            bbox_transform=ax.transAxes,
        )
        _ = df_adm[["subject", "stream", "bias", "kappa"]].melt(
            id_vars=["subject", "stream"], var_name="parameter"
        )
        order = ["kappa", "bias"]
        sns.barplot(
            x="parameter", order=order, y="value", hue="stream", data=_, ci=68, ax=axins
        )
        axins.set(ylabel="", xlabel="")
        axins.get_legend().remove()
        axins.axhline(1, xmax=0.5, **axhline_args)
        sns.despine(ax=axins)
        axins.set_xticklabels([i.capitalize() for i in order])

    draw_k1_d = False
    if draw_k1_d:
        meanb = df_adm["bias"].mean()
        vals = np.abs(eq1(numbers_rescaled, bias=meanb, kappa=1))
        ax.plot(
            NUMBERS - 1,
            vals,
            color="black",
            ls="--",
            zorder=10,
            label=f"b={meanb:.2f}, k=1",
        )

    sns.despine(ax=ax)
    legend_d = False
    if legend_d:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            title=None,
            handles=handles,
            labels=[i.capitalize() if not i.startswith("b") else i for i in labels],
            frameon=False,
        )

    ax.set(ylabel=r"$|dv|$", xlabel="Sample")

    if use_smalltitle:
        ax.text(
            x=0.5,
            y=0.9,
            s="Model",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.set_title("Model", fontweight="bold")

# %%
# Final settings and save
fig.savefig(fname_fig4, bbox_inches="tight")

# %%
