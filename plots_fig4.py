"""Figure 4 plot - CPP/P3."""
# %%
# Imports
import matplotlib.pyplot as plt
import mne
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import ANALYSIS_DIR_LOCAL, NUMBERS, STREAMS

# %%
# Settings
analysis_dir = ANALYSIS_DIR_LOCAL

axhline_args = dict(color="black", linestyle="--", linewidth=1)

# %%
# Prepare file paths
fname_fig4 = analysis_dir / "figures" / "fig4_pre.pdf"

fname_erps = analysis_dir / "derived_data" / "erps.tsv"
fname_amps = analysis_dir / "derived_data" / "erp_amps.tsv"

# %%
# Load data
df_erps = pd.read_csv(fname_erps, sep="\t")
df_amps = pd.read_csv(fname_amps, sep="\t")

baseline = eval(df_amps["baseline"][0])
mean_times = eval(df_amps["mean_times"][0])
p3_group = eval(df_amps["p3_group"][0])

# prepare MNE montage
montage = mne.channels.make_standard_montage("easycap-M1")
info = mne.create_info(p3_group, 1, ch_types="eeg")
info.set_montage(montage)

# %%
# Start figure
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.tight_layout()

# %%
# plot ERPs
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
            ci=None,
            ax=ax,
            palette=palette,
        )

        sns.despine(ax=ax)
        ylabel = "Amplitude (µV)" if istream == 0 else ""
        ax.set(xlabel="Time (s)", ylabel=ylabel)
        ax.axhline(0, **axhline_args)
        ax.axvline(0, **axhline_args)

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

    for ax in axs[0, ...]:
        ax.set_ylim(
            (
                min(axs[0, 0].get_ylim()[0], axs[0, 1].get_ylim()[0]),
                max(axs[0, 0].get_ylim()[1], axs[0, 1].get_ylim()[1]),
            )
        )

# %%
# plot mean amplitudes
pass


# %%
# switch axis off (placeholder)
axs[1, 1].axis("off")

# %%
# Final settings and save
fig.savefig(fname_fig4, bbox_inches="tight")

# %%
