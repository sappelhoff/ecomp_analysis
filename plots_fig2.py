"""Figure 2 plot - RSA models and timecourses."""
# %%
# Imports
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ANALYSIS_DIR_LOCAL, STREAMS
from model_rdms import get_models_dict

# %%
# Settings
analysis_dir = ANALYSIS_DIR_LOCAL
rdm_size = "18x18"

axhline_args = dict(color="black", linestyle="--", linewidth=1)

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
    fig.tight_layout()
# %%
# Plot model RDMS

with sns.plotting_context("poster"):
    for model, panel in zip(modelnames, ("a", "b", "c")):
        toplot = models_dict["no_orth"][model]

        axd[panel].imshow(toplot, cmap="viridis")

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
rsa_colors = {
    "digit": "C0",
    "color": "C3",
    "numberline": "C4",
}

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
            ci=None,
            ax=ax,
            palette=rsa_colors,
        )

        if stream in STREAMS:
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

        ax.axhline(0, **axhline_args)
        ax.axvline(0, **axhline_args)
        assert np.unique(data["method"])[0] == "pearson"
        ax.set(ylabel="Pearson's r", xlabel="Time (s)")
        ax.axvspan(*window_sel, color="black", alpha=0.1)
        sns.despine(ax=ax)

        # plot significance bars
        for model in modelnames:
            y = ax.get_ylim()[0]
            clusters = permdistr_dict[model][stream]["sig_clusters"]
            if len(clusters) == 0:
                continue
            for clu in clusters:
                ax.plot(times[clu], [y] * len(clu), c=rsa_colors[model], ls="-")


# %%
# Final settings and save
fig.savefig(fname_fig2, bbox_inches="tight")

# %%
