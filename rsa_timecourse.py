"""Calculate RSA timecourse.

- create numberline model RDM
- For each subj and stream:
    - load rdm_times array
    - for each timepoint
        - correlate ERP-RDM with model RDM
- plot means over subjs for each stream
- check which timewindow is has high correlation
- plot mean ERP-RDMs for each stream over subjs and selected timewindow

"""
# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL, SUBJS
from utils import calc_rdm, prep_to_plot, rdm2vec

# %%
# Settings
# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

rsa_method = "pearson"
distance_measure = "mahalanobis"

numbers = np.arange(1, 10, dtype=int)
streams = ["single", "dual"]

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"

mahal_dir = data_dir / "derivatives" / "rsa" / "rdms_mahalanobis"
mahal_dir.mkdir(exist_ok=True, parents=True)

fname_rdm_template = str(mahal_dir / "sub-{:02}_stream-{}_rdm-mahal.npy")
fname_times = mahal_dir / "times.npy"

# %%
# Get times for RDM timecourses
times = np.load(fname_times)

# %%
# Calculate model RDM
numberline = calc_rdm(numbers, normalize=True)

# %%
# Calculate RSA per subj and stream
df_rsa_list = []
rdm_times_streams_subjs = np.full(
    (len(numbers), len(numbers), len(times), len(streams), len(SUBJS)), np.nan
)
for isub, sub in enumerate(tqdm(SUBJS)):
    for istream, stream in enumerate(streams):
        rdm_times = np.load(fname_rdm_template.format(sub, stream))
        rdm_times_streams_subjs[..., istream, isub] = rdm_times

        # Correlation
        ntimes = rdm_times.shape[-1]
        x = rdm2vec(numberline, lower_tri=True)
        corr_model_times = np.full((ntimes, 1), np.nan)
        for itime in range(ntimes):
            y = rdm2vec(rdm_times[..., itime], lower_tri=True)
            if rsa_method == "pearson":
                corr, _ = scipy.stats.pearsonr(x, y)
            else:
                raise RuntimeError(f"invalid rsa_method: {rsa_method}")
            corr_model_times[itime, 0] = corr

        # Make a dataframe
        _df_rsa = pd.DataFrame(corr_model_times, columns=["similarity"])
        _df_rsa.insert(0, "model", "numberline")
        _df_rsa.insert(0, "method", rsa_method)
        _df_rsa.insert(0, "measure", distance_measure)
        _df_rsa.insert(0, "itime", range(ntimes))
        _df_rsa.insert(0, "time", times)
        _df_rsa.insert(0, "stream", stream)
        _df_rsa.insert(0, "subject", sub)

        # Save
        df_rsa_list.append(_df_rsa)

df_rsa = pd.concat(df_rsa_list)
df_rsa = df_rsa.reset_index(drop=True)
assert len(df_rsa) == ntimes * len(SUBJS) * len(streams)
assert not np.isnan(rdm_times_streams_subjs).any()
df_rsa

# %%
# Plot the data
ylabel = {"pearson": "Pearson's r"}[rsa_method]

window_sel = (0.2, 0.6)  # representative window, look at figure

with sns.plotting_context("talk"):
    fig, ax = plt.subplots()

    sns.lineplot(data=df_rsa, x="time", y="similarity", hue="stream", ci=68, ax=ax)
    ax.axhline(0, color="black", lw=0.25, ls="--")
    ax.set(ylabel=f"{ylabel}", xlabel="Time (s)")
    ax.axvspan(*window_sel, color="black", alpha=0.1)
    sns.despine(fig)

# %%
# Show mean RDMs in selected timewindow
idx_start, idx_stop = np.unique(df_rsa[df_rsa["time"].isin(window_sel)]["itime"])

rdm_times_streams_subjs.shape

mean_rdms = {}
for istream, stream in enumerate(streams):
    # mean over subjects
    rdm_submean = np.mean(rdm_times_streams_subjs[..., istream, :], axis=-1)

    # mean over time
    rdm_subtimemean = np.mean(rdm_submean[..., idx_start:idx_stop], axis=-1)
    mean_rdms[stream] = rdm_subtimemean


with sns.plotting_context("talk"):
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        sharex=True,
        sharey=True,
    )

    for i, stream in enumerate(streams):
        ax = axs.flat[i]
        plotrdm = prep_to_plot(mean_rdms[stream])

        im = ax.imshow(plotrdm)
        fig.colorbar(
            im,
            ax=ax,
            label=f"{distance_measure}",
            orientation="horizontal",
            shrink=0.75,
        )

        ax.set_title(stream)
        ax.xaxis.set_major_locator(plt.MaxNLocator(len(numbers)))
        ax.set_xticklabels([""] + [str(j) for j in numbers])
        ax.yaxis.set_major_locator(plt.MaxNLocator(len(numbers)))
        ax.set_yticklabels([""] + [str(j) for j in numbers])

    fig.suptitle(
        f"RDMs, mean over subjects and time\n({window_sel[0]}s - {window_sel[1]}s)",
        y=1.1,
    )

# %%
