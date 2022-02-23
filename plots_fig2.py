"""Figure 2 plot - RSA models and timecourses."""
# %%
# Imports
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import ANALYSIS_DIR_LOCAL, STREAMS
from model_rdms import get_models_dict

# %%
# Settings
analysis_dir = ANALYSIS_DIR_LOCAL
rdm_size = "18x18"

axhline_args = dict(color="black", linestyle="--", linewidth=1)

rsa_colors = {
    "digit": "C2",
    "color": "C4",
    "numberline": "C9",
}

# %%
# File paths
fname_rsa = analysis_dir / "derived_data" / "rsa_timecourses.tsv"
fname_perm = analysis_dir / "derived_data" / "rsa_perm_results.json"

fname_fig2 = analysis_dir / "figures" / "fig2.pdf"


# %%
# Get model RDMs
modelnames = ["digit", "color", "numberline"]
nmodels = len(modelnames)
models_dict = get_models_dict(rdm_size, modelnames, orth=False)

# %%
with sns.plotting_context("poster"):
    mosaic = """
    aaaabbbbcccc
    aaaabbbbcccc
    aaaabbbbcccc
    aaaabbbbcccc
    ddddddeeeeee
    ddddddeeeeee
    ddddddeeeeee
    ddddddeeeeee
    ...ffffff...
    ...ffffff...
    ...ffffff...
    ...ffffff...
    """
    fig = plt.figure(figsize=(20, 20))
    axd = fig.subplot_mosaic(mosaic)
    fig.tight_layout(h_pad=3, w_pad=2)
# %%
# Plot model RDMS

with sns.plotting_context("poster"):
    for model, panel in zip(modelnames, ("a", "b", "c")):
        toplot = models_dict["no_orth"][model]
        ax = axd[panel]
        im = ax.imshow(toplot, cmap="viridis")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)

        if panel != "c":
            cax.axis("off")
        else:
            cbar = plt.colorbar(im, cax=cax)

        ax.set_title(model.capitalize(), color=rsa_colors[model], fontweight="bold")
        ax.xaxis.set_major_locator(plt.MaxNLocator(18))
        ax.yaxis.set_major_locator(plt.MaxNLocator(18))

        xy_ticklabels = [ax.get_xticklabels(), ax.get_yticklabels()]
        for ticklabels in xy_ticklabels:
            for itick, tick in enumerate(ticklabels):
                color = "red"
                count = itick
                if itick > 9:
                    color = "blue"
                    count -= 9
                tick.set_color(color)
                tick.set_text(str(count))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="FixedFormatter .* FixedLocator",
            )
            ax.set_xticklabels(ax.get_xticklabels())
            ax.set_yticklabels(ax.get_yticklabels())


# %%
# plot timecourses
window_sel = (0.2, 0.6)
df_rsa = pd.read_csv(fname_rsa, sep="\t")
times = np.unique(df_rsa["time"])
with open(fname_perm, "r") as fin:
    permdistr_dict = json.load(fin)
sortby = [
    "subject",
    "stream",
    "orth",
    "model",
    "time",
    "itime",
    "measure",
    "method",
    "rdm_size",
]
df_rsa = df_rsa.sort_values(by=sortby)[sortby + ["similarity"]]

with sns.plotting_context("poster"):
    for panel, stream in zip(("d", "e", "f"), ["single", "dual", "diff"]):
        ax = axd[panel]

        if stream in STREAMS:
            data = df_rsa[(df_rsa["orth"]) & (df_rsa["stream"] == stream)]
        else:
            assert stream == "diff"
            single = df_rsa[(df_rsa["stream"] == "single")]["similarity"]
            dual = df_rsa[(df_rsa["stream"] == "dual")]["similarity"]
            df_diff = (
                df_rsa[(df_rsa["stream"] == "single")].copy().reset_index(drop=True)
            )
            df_diff["similarity"] = dual.to_numpy() - single.to_numpy()
            df_diff["stream"] = "dual-single"
            data = df_diff[df_diff["orth"]]

        sns.lineplot(
            data=data,
            x="time",
            y="similarity",
            hue="model",
            hue_order=modelnames,
            ci=68,
            ax=ax,
            palette=rsa_colors,
        )

        # legend
        if stream in STREAMS:
            ax.get_legend().remove()
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1, 1),
                frameon=False,
                handles=handles,
                labels=[lab.capitalize() for lab in labels],
            )

        ax.axhline(0, **axhline_args)
        ax.axvline(0, **axhline_args)
        assert np.unique(data["method"])[0] == "pearson"
        ax.set(ylabel="Pearson's r", xlabel="Time (s)")
        ax.axvspan(*window_sel, color="black", alpha=0.1)
        sns.despine(ax=ax)

        # title
        ax.text(
            x=0.5,
            y=0.95,
            s=stream.capitalize() if stream != "diff" else "Dual - Single",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        # plot significance bars
        for model in modelnames:
            y = ax.get_ylim()[0]
            clusters = permdistr_dict[model][stream]["sig_clusters"]
            if len(clusters) == 0:
                continue
            for clu in clusters:
                ax.plot(times[clu], [y] * len(clu), c=rsa_colors[model], ls="-")

    # adjust ylims
    ylims = (
        min(axd["d"].get_ylim()[0], axd["e"].get_ylim()[0]),
        max(axd["d"].get_ylim()[1], axd["e"].get_ylim()[1]),
    )
    axd["d"].set_ylim(ylims)
    axd["e"].set_ylim(ylims)
    axd["f"].set_ylim((axd["f"].get_ylim()[0], axd["f"].get_ylim()[1] * 1.1))

# %%
# Final settings and save
fig.savefig(fname_fig2, bbox_inches="tight")

# %%
