"""Calculate RSA timecourse.

- create model RDMs
- For each subj and stream:
    - load rdm_times array
    - for each timepoint
        - correlate ERP-RDM with model RDMs
- plot means over subjs for each stream
- check which timewindow is has high correlation
- plot mean ERP-RDMs for each stream over subjs and selected time window

"""
# %%
# Imports
import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from clusterperm import (
    _perm_X_paired,
    get_max_stat,
    get_significance,
    perm_X_1samp,
    prep_for_clusterperm,
    return_clusters,
)
from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL, NUMBERS, STREAMS, SUBJS
from utils import calc_rdm, prep_to_plot, spm_orth

# %%
# Settings
# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

rsa_method = "pearson"
distance_measure = "mahalanobis"
rdm_size = "18x18"  # "9x9" or "18x18"

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"

mahal_dir = data_dir / "derivatives" / "rsa" / rdm_size / "rdms_mahalanobis"

fname_rdm_template = str(mahal_dir / "sub-{:02}_stream-{}_rdm-mahal.npy")
fname_times = mahal_dir / "times.npy"

# %%
# Get times for RDM timecourses
times = np.load(fname_times)

# %%
# Calculate model RDMs
ndim = len(NUMBERS)
if rdm_size == "9x9":
    conditions = NUMBERS
else:
    assert rdm_size == "18x18"
    conditions = np.hstack((NUMBERS, NUMBERS))

model_numberline = calc_rdm(conditions, normalize=True)

model_extremity = calc_rdm(np.abs(conditions - 5), normalize=True)

# following models are only 18x18
model_color = np.repeat(
    np.repeat(np.abs(np.identity(2) - 1), ndim, axis=1), ndim, axis=0
)

model_digit = np.tile(np.abs(np.identity(ndim) - 1), (2, 2))

model_parity = np.tile(np.tile(np.abs(np.identity(2) - 1), ndim).T, ndim)

model_numXcat = np.vstack(
    (
        np.hstack(
            (
                calc_rdm(NUMBERS, normalize=True),
                np.fliplr(calc_rdm(NUMBERS, normalize=True)),
            )
        ),
        np.hstack(
            (
                np.fliplr(calc_rdm(NUMBERS, normalize=True)),
                calc_rdm(NUMBERS, normalize=True),
            )
        ),
    )
)

# Include or exclude models
if rdm_size == "9x9":
    models_dict = {
        "no_orth": {"numberline": model_numberline, "extremity": model_extremity}
    }
else:
    assert rdm_size == "18x18"
    take_all = False
    if take_all:
        models_dict = {
            "no_orth": {
                "digit": model_digit,
                "color": model_color,
                "parity": model_parity,
                "numberline": model_numberline,
                "numXcat": model_numXcat,
                "extremity": model_extremity,
            }
        }
    else:
        models_dict = {
            "no_orth": {
                "digit": model_digit,
                "color": model_color,
                "numberline": model_numberline,
            }
        }

nmodels = len(models_dict["no_orth"])

# %%
# Orthogonalize models
# First make a copy of non-orth models, this copy will be modified below
models_dict["orth"] = copy.deepcopy(models_dict["no_orth"])

# orthogonalize recursively using spm_orth
# The last column vector in the output of the spm_orth function
# is orthogonalized with respect to all previous column vectors.
#
# Iterate through models, for each model, obtain its orthed version
# by placing it in the last column in a matrix X.
# For example, for models 1, 2, 3, 4, 5 do the following column orders:
#
# 2 3 4 5 1
# 1 3 4 5 2
# 1 2 4 5 3
# 1 2 3 5 4
# 1 2 3 4 5
#
# ... to obtain the orthed models (5 calls to spm_orth needed).
#
#  NOTE: Each column must be mean-centered, and only the vector
#        form of a square symmetric RDM should be passed to spm_orth
#        That is, use "squareform", or extract the lower (or upper)
#        triangle of the RDM, excluding the diagonal (but do not mix
#        these approaches).
model_arrs = np.stack(list(models_dict["orth"].values()), axis=-1)
modelnames = list(models_dict["orth"].keys())
imodels = np.arange(nmodels)
orthmodels = []
for imodel in imodels:
    orth_col_order = np.hstack((imodels[imodels != imodel], imodel)).tolist()
    X = np.full((len(squareform(model_numberline)), nmodels), np.nan)
    for imod in range(nmodels):
        icol = orth_col_order.index(imod)
        vec = squareform(model_arrs[..., imod])  # convert to condensed vector
        X[..., icol] = vec - vec.mean()  # mean-center before orth
    X_orth = spm_orth(X)
    orth_model = squareform(X_orth[..., -1])  # convert to square symmetric RDM again
    orthmodels.append((modelnames[imodel], orth_model))

# update copied dict with final, orthogonalized models
for modelname, orthmodel in orthmodels:
    models_dict["orth"][modelname] = orthmodel


# %%
# plot models
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(
        2, nmodels, sharex=True, sharey=True, figsize=(4 * nmodels, 6)
    )

    for imodel in range(nmodels):

        model = list(models_dict["no_orth"].values())[imodel]
        orthmodel = list(models_dict["orth"].values())[imodel]
        ax1, ax2 = axs[:, imodel]
        im1 = ax1.imshow(model, vmin=0, vmax=1, cmap="viridis")
        im2 = ax2.imshow(orthmodel, vmin=-1, vmax=1, cmap="viridis")
        ax1.set_title(modelnames[imodel])
        ax2.set_title("orth " + modelnames[imodel])
        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)

fig.tight_layout()

# %%
# Calculate RSA per subj and stream
df_rsa_list = []
rdm_times_streams_subjs = np.full(
    (len(conditions), len(conditions), len(times), len(STREAMS), len(SUBJS)), np.nan
)
for orth_str, models in models_dict.items():
    orth = dict(no_orth=False, orth=True)[orth_str]

    for imodel, (modelname, model) in enumerate(tqdm(models.items())):
        for isub, sub in enumerate(SUBJS):
            for istream, stream in enumerate(STREAMS):
                rdm_times = np.load(fname_rdm_template.format(sub, stream))
                rdm_times_streams_subjs[..., istream, isub] = rdm_times

                # Correlation with model
                ntimes = rdm_times.shape[-1]
                x = squareform(model)
                corr_model_times = np.full((ntimes, 1), np.nan)
                for itime in range(ntimes):
                    y = squareform(rdm_times[..., itime])
                    if rsa_method == "pearson":
                        corr, _ = scipy.stats.pearsonr(x, y)
                    else:
                        raise RuntimeError(f"invalid rsa_method: {rsa_method}")
                    corr_model_times[itime, 0] = corr

                # Make a dataframe
                _df_rsa = pd.DataFrame(corr_model_times, columns=["similarity"])
                _df_rsa.insert(0, "rdm_size", rdm_size)
                _df_rsa.insert(0, "orth", orth)
                _df_rsa.insert(0, "model", modelname)
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
assert len(df_rsa) == ntimes * len(SUBJS) * len(STREAMS) * nmodels * 2
assert not np.isnan(rdm_times_streams_subjs).any()
df_rsa

# %%
# Cluster based permutation testing
test_models = ["digit", "color", "numberline"]
test_orth = True
clusterstat = "length"
thresh = 0.001
clusterthresh = 0.001
niterations = 1000
ttest_kwargs = dict(axis=0, nan_policy="raise", alternative="two-sided")

sig_clusters_dict = {
    mod: {"1samp_single": [], "1samp_dual": [], "paired": []} for mod in test_models
}
print(
    f"Running permutation testing for {len(test_models)} models, "
    "1-sample and paired t-tests each."
)
for test_model in test_models:
    X = prep_for_clusterperm(df_rsa, test_model, test_orth)
    nsubjs, ntimes, nstreams = X.shape

    # Paired ttest
    # ---------------------------------------------------------------------------------
    # Get permutation distribution
    distr = np.full(niterations, np.nan)
    for iteration in tqdm(range(niterations)):
        # Permute data
        Xperm = _perm_X_paired(X, nsubjs, ntimes)

        # Calculate statistics
        tval, pval = scipy.stats.ttest_rel(Xperm[..., 0], Xperm[..., 1], **ttest_kwargs)

        # Find clusters and cluster statistic
        clusters = return_clusters(pval < thresh)
        distr[iteration] = get_max_stat(clusters, clusterstat, tval)

    # calculate observed stats and evaluate significance
    tval_obs, pval_obs = scipy.stats.ttest_rel(X[..., 0], X[..., 1], **ttest_kwargs)
    clusters_obs = return_clusters(pval_obs < thresh)
    _, sig_clusters, _ = get_significance(
        distr, clusterstat, clusters_obs, tval_obs, clusterthresh
    )
    sig_clusters_dict[test_model]["paired"] += sig_clusters

    # One samp ttests
    # ---------------------------------------------------------------------------------
    # Get permutation distribution
    distr0 = np.full(niterations, np.nan)
    distr1 = np.full(niterations, np.nan)
    for iteration in tqdm(range(niterations)):
        # Permute data
        Xperm = perm_X_1samp(X, nsubjs, ntimes, nstreams)

        # Calculate statistics
        tval0, pval0 = scipy.stats.ttest_1samp(Xperm[..., 0], popmean=0)
        tval1, pval1 = scipy.stats.ttest_1samp(Xperm[..., 1], popmean=0)

        # Find clusters and cluster statistic
        clusters0 = return_clusters(pval0 < thresh)
        clusters1 = return_clusters(pval1 < thresh)
        distr0[iteration] = get_max_stat(clusters0, clusterstat, tval0)
        distr1[iteration] = get_max_stat(clusters1, clusterstat, tval1)

    # calculate observed stats and evaluate significance
    tval_obs0, pval_obs0 = scipy.stats.ttest_1samp(X[..., 0], popmean=0)
    tval_obs1, pval_obs1 = scipy.stats.ttest_1samp(X[..., 1], popmean=0)
    clusters_obs0 = return_clusters(pval_obs0 < thresh)
    clusters_obs1 = return_clusters(pval_obs1 < thresh)
    _, sig_clusters0, _ = get_significance(
        distr0, clusterstat, clusters_obs0, tval_obs0, clusterthresh
    )
    _, sig_clusters1, _ = get_significance(
        distr1, clusterstat, clusters_obs1, tval_obs1, clusterthresh
    )
    sig_clusters_dict[test_model]["1samp_single"] += sig_clusters0
    sig_clusters_dict[test_model]["1samp_dual"] += sig_clusters1

# %%
# Plot the data
ylabel = {"pearson": "Pearson's r"}[rsa_method]
rsa_colors = {
    "digit": "C0",
    "color": "C3",
    "numberline": "C4",
    "extremity": "C5",
    "parity": "C1",
    "numXcat": "C2",
}
min_clu_len = 0
min_clu_len_ms = ((1 / 250) * min_clu_len) * 1000

# representative windows, look at figure
window_sels = [(0.075, 0.19), (0.21, 0.6)]

with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    for iax, ax in enumerate(axs.flat):

        data = df_rsa[df_rsa["orth"] == bool(iax)]

        sns.lineplot(
            data=data,
            x="time",
            y="similarity",
            hue="model",
            style="stream",
            ci=68,
            ax=ax,
            palette=rsa_colors,
        )
        ax.axhline(0, color="black", lw=0.25, ls="--")
        ax.axvline(0, color="black", lw=0.25, ls="--")
        ax.set(ylabel=f"{ylabel}", xlabel="Time (s)")
        for window_sel in window_sels:
            ax.axvspan(*window_sel, color="black", alpha=0.1)

        if iax == 0:
            ax.get_legend().remove()
        else:
            sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))

        ax.set_title(f"orth = {bool(iax)}")

        # plot significance bars (if present)
        if bool(iax) != test_orth:
            continue
        skipped_clus = []
        y = ax.get_ylim()[0]
        for test_model in test_models:
            c = rsa_colors[test_model]
            for test, clusters in sig_clusters_dict[test_model].items():
                if len(clusters) == 0:
                    continue
                ls = {"1samp_single": "-", "1samp_dual": "--", "paired": ":"}[test]
                if any([len(clu) > min_clu_len for clu in clusters]):
                    y -= 0.02
                for clu in clusters:
                    if len(clu) <= min_clu_len:
                        skipped_clus += [test_model + "_" + test]
                        continue
                    ax.plot(times[clu], [y] * len(clu), c=c, ls=ls)
        print(
            f"{len(skipped_clus)} clusters shorter than {min_clu_len_ms} ms not shown:"
            f"\n{skipped_clus}"
        )

sns.despine(fig)
fig.tight_layout()

# %%
# Show mean RDMs in selected time windows
for window_sel in window_sels:

    # find measured time closest to our selection
    start = times[np.argmin(np.abs(times - window_sel[0]))]
    stop = times[np.argmin(np.abs(times - window_sel[1]))]

    idx_start, idx_stop = np.unique(df_rsa[df_rsa["time"].isin((start, stop))]["itime"])

    rdm_times_streams_subjs.shape

    mean_rdms = {}
    for istream, stream in enumerate(STREAMS):
        # mean over subjects
        rdm_submean = np.mean(rdm_times_streams_subjs[..., istream, :], axis=-1)

        # mean over time
        rdm_subtimemean = np.mean(rdm_submean[..., idx_start:idx_stop], axis=-1)
        mean_rdms[stream] = rdm_subtimemean

    with sns.plotting_context("talk"):
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(10, 8),
            sharex=True,
            sharey=True,
        )

        for i, stream in enumerate(STREAMS):
            ax = axs.flat[i]
            plotrdm = prep_to_plot(mean_rdms[stream])

            im = ax.imshow(plotrdm)
            fig.colorbar(
                im,
                ax=ax,
                label=f"{distance_measure}",
                orientation="horizontal",
                shrink=0.75,
                pad=0.25,
            )

            ax.set_title(stream)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="FixedFormatter .* FixedLocator",
                )
                if rdm_size == "9x9":
                    ax.xaxis.set_major_locator(plt.MaxNLocator(len(NUMBERS)))
                    ax.set_xticklabels([""] + [str(j) for j in NUMBERS])
                    ax.yaxis.set_major_locator(plt.MaxNLocator(len(NUMBERS)))
                    ax.set_yticklabels([""] + [str(j) for j in NUMBERS])
                else:
                    assert rdm_size == "18x18"
                    ax.xaxis.set_major_locator(plt.MaxNLocator(len(NUMBERS) * 2))
                    ax.set_xticklabels([""] + [str(j) for j in NUMBERS] * 2)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(len(NUMBERS) * 2))
                    ax.set_yticklabels([""] + [str(j) for j in NUMBERS] * 2)
                    ax.xaxis.set_tick_params(labelsize=10)
                    ax.yaxis.set_tick_params(labelsize=10)
                    ax.set_xlabel("red             blue", labelpad=12)
                    ax.set_ylabel("blue             red", labelpad=12)

        fig.suptitle(
            f"RDMs, mean over subjects and time\n({window_sel[0]}s - {window_sel[1]}s)",
            y=1.1,
        )

# %%
