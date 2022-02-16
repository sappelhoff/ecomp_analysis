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
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from clusterperm import (
    get_max_stat,
    get_significance,
    perm_X_1samp,
    prep_for_clusterperm,
    return_clusters,
)
from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL, NUMBERS, STREAMS, SUBJS
from model_rdms import get_models_dict
from utils import prep_to_plot

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
# Get model RDMs
modelnames = ["digit", "color", "numberline"]
nmodels = len(modelnames)
models_dict = get_models_dict(rdm_size, modelnames, bias=None, kappa=None)

# %%
# Try an compressed versus anticompressed numberline model
comp_anticomp_timecourse = True
if comp_anticomp_timecourse:
    kcomp = 0.1
    kacomp = 10
    models_dict_comp = get_models_dict(rdm_size, modelnames, bias=0, kappa=kcomp)
    models_dict_acomp = get_models_dict(rdm_size, modelnames, bias=0, kappa=kacomp)

    models_dict = {"no_orth": {}, "orth": {}}
    models_dict["no_orth"][f"numberline_k-{kcomp}"] = models_dict_comp["no_orth"][
        "numberline"
    ]
    models_dict["no_orth"][f"numberline_k-{kacomp}"] = models_dict_acomp["no_orth"][
        "numberline"
    ]
    models_dict["orth"][f"numberline_k-{kcomp}"] = models_dict_comp["orth"][
        "numberline"
    ]
    models_dict["orth"][f"numberline_k-{kacomp}"] = models_dict_acomp["orth"][
        "numberline"
    ]

    nmodels = 2
    modelnames = [
        f"numberline_k-{kcomp}",
        f"numberline_k-{kacomp}",
    ]

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
nconditions = int(rdm_size.split("x")[0])
df_rsa_list = []
rdm_times_streams_subjs = np.full(
    (nconditions, nconditions, len(times), len(STREAMS), len(SUBJS)), np.nan
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
test_models = modelnames
test_orth = True
clusterstat = "length"
thresh = 0.01
clusterthresh = 0.01
niterations = 1000
ttest_kwargs = dict(axis=0, nan_policy="raise", alternative="two-sided")

tests = ["1samp_single", "1samp_dual", "paired"]
permdistr_dict = {
    mod: {"1samp_single": {}, "1samp_dual": {}, "paired": {}} for mod in test_models
}
print(f"Running permutation testing for {len(test_models)} models")

# Paired ttest is run as a 1-sample ttest of the difference between conditions
# (this is equivalent)
rng = np.random.default_rng(1337)
dfs = []
for test_model in test_models:
    X = prep_for_clusterperm(df_rsa, test_model, test_orth)
    X_paired = X[..., 0] - X[..., 1]

    # Get permutation distributions
    distr0 = np.full(niterations, np.nan)
    distr1 = np.full(niterations, np.nan)
    distr_paired = np.full(niterations, np.nan)
    for iteration in tqdm(range(niterations)):
        # Permute data
        Xperm = perm_X_1samp(X, rng)
        Xperm_paired = perm_X_1samp(X_paired, rng)

        # Calculate statistics
        tval0, pval0 = scipy.stats.ttest_1samp(Xperm[..., 0], popmean=0, **ttest_kwargs)
        tval1, pval1 = scipy.stats.ttest_1samp(Xperm[..., 1], popmean=0, **ttest_kwargs)
        tval_paired, pval_paired = scipy.stats.ttest_1samp(
            Xperm_paired, popmean=0, **ttest_kwargs
        )

        # Find clusters and cluster statistic
        clusters0 = return_clusters(pval0 < thresh)
        clusters1 = return_clusters(pval1 < thresh)
        clusters_paired = return_clusters(pval_paired < thresh)
        distr0[iteration] = get_max_stat(clusters0, clusterstat, tval0)
        distr1[iteration] = get_max_stat(clusters1, clusterstat, tval1)
        distr_paired[iteration] = get_max_stat(
            clusters_paired, clusterstat, tval_paired
        )

    # collect permutation distributions
    df_distr = pd.DataFrame([distr0, distr1, distr_paired]).T
    df_distr.columns = tests
    df_distr["model"] = test_model
    dfs.append(df_distr)

    # calculate observed stats and evaluate significance
    tval_obs0, pval_obs0 = scipy.stats.ttest_1samp(X[..., 0], popmean=0, **ttest_kwargs)
    tval_obs1, pval_obs1 = scipy.stats.ttest_1samp(X[..., 1], popmean=0, **ttest_kwargs)
    tval_obs_paired, pval_obs_paired = scipy.stats.ttest_1samp(
        X_paired, popmean=0, **ttest_kwargs
    )
    clusters_obs0 = return_clusters(pval_obs0 < thresh)
    clusters_obs1 = return_clusters(pval_obs1 < thresh)
    clusters_obs_paired = return_clusters(pval_obs_paired < thresh)
    (
        permdistr_dict[test_model]["1samp_single"]["clusterthresh_stat"],
        permdistr_dict[test_model]["1samp_single"]["cluster_stats"],
        permdistr_dict[test_model]["1samp_single"]["sig_clusters"],
        permdistr_dict[test_model]["1samp_single"]["pvals"],
    ) = get_significance(distr0, clusterstat, clusters_obs0, tval_obs0, clusterthresh)
    (
        permdistr_dict[test_model]["1samp_dual"]["clusterthresh_stat"],
        permdistr_dict[test_model]["1samp_dual"]["cluster_stats"],
        permdistr_dict[test_model]["1samp_dual"]["sig_clusters"],
        permdistr_dict[test_model]["1samp_dual"]["pvals"],
    ) = get_significance(distr1, clusterstat, clusters_obs1, tval_obs1, clusterthresh)
    (
        permdistr_dict[test_model]["paired"]["clusterthresh_stat"],
        permdistr_dict[test_model]["paired"]["cluster_stats"],
        permdistr_dict[test_model]["paired"]["sig_clusters"],
        permdistr_dict[test_model]["paired"]["pvals"],
    ) = get_significance(
        distr_paired, clusterstat, clusters_obs_paired, tval_obs_paired, clusterthresh
    )

df_distr = pd.concat(dfs)

# %%
# Plot permutation distributions
g = sns.displot(
    data=df_distr.melt(
        id_vars=["model"], var_name="distribution", value_name="max_stat"
    ),
    x="max_stat",
    row="model",
    col="distribution",
    kind="kde",
    cut=0,
    facet_kws=dict(sharex=False, sharey=False),
)

for (row_val, col_val), ax in g.axes_dict.items():
    clusterthresh_stat = permdistr_dict[row_val][col_val]["clusterthresh_stat"]
    ax.axvline(clusterthresh_stat, color="blue", zorder=0)

    cluster_stats = permdistr_dict[row_val][col_val]["cluster_stats"]
    for stat in cluster_stats:
        ax.axvline(stat, color="red", ls="--", zorder=1)


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
if comp_anticomp_timecourse:
    rsa_colors = {m: f"C{i}" for i, m in enumerate(modelnames)}
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
            for test in tests:
                clusters = permdistr_dict[test_model][test]["sig_clusters"]
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
# Look at single subject RSA time courses
g = sns.relplot(
    kind="line",
    data=data,
    x="time",
    y="similarity",
    hue="model",
    style="stream",
    ci=None,
    palette=rsa_colors,
    col="subject",
    col_wrap=6,
)

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
