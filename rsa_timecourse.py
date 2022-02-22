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
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
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

# comparing compression and anti-compression
comp_anticomp_timecourse = False

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"

mahal_dir = data_dir / "derivatives" / "rsa" / rdm_size / "rdms_mahalanobis"

fname_rdm_template = str(mahal_dir / "sub-{:02}_stream-{}_rdm-mahal.npy")
fname_times = mahal_dir / "times.npy"

fname_rsa = analysis_dir / "derived_data" / "rsa_timecourses.tsv"
fname_perm = analysis_dir / "derived_data" / "rsa_perm_results.json"
# %%
# Get times for RDM timecourses
times = np.load(fname_times)

# %%
# Get model RDMs
modelnames = ["digit", "color", "numberline"]
nmodels = len(modelnames)
models_dict = get_models_dict(rdm_size, modelnames, True, bias=None, kappa=None)

# %%
# Try an compressed versus anticompressed numberline model
if comp_anticomp_timecourse:
    kcomp = 0.1
    kacomp = 10
    models_dict_comp = get_models_dict(rdm_size, modelnames, True, bias=0, kappa=kcomp)
    models_dict_acomp = get_models_dict(
        rdm_size, modelnames, True, bias=0, kappa=kacomp
    )

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

df_rsa = pd.concat(df_rsa_list).reset_index(drop=True)
df_rsa = df_rsa.reset_index(drop=True)
assert len(df_rsa) == ntimes * len(SUBJS) * len(STREAMS) * nmodels * 2
assert not np.isnan(rdm_times_streams_subjs).any()

# %%
# Save RSA results
df_rsa.to_csv(fname_rsa, sep="\t", na_rep="n/a", index=False)

# %%
# Cluster based permutation testing
# Run 1-sample ttests for each model and stream
# Run paired ttests (single vs. dual) for each model, which is equivalent
# to a 1-sample ttest of the difference between single and dual
test_models = modelnames
test_orth = True
clusterstat = "length"
thresh = 0.01
clusterthresh = 0.01
niterations = 1000
ttest_kwargs = dict(axis=0, nan_policy="raise", alternative="two-sided", popmean=0)

tests = ["single", "dual", "diff"]
permdistr_dict = {mod: {"single": {}, "dual": {}, "diff": {}} for mod in test_models}
print(f"Running {len(tests)} cluster permutation tests for {len(test_models)} models")

rng = np.random.default_rng(1337)
dfs = []
for test_model in test_models:
    X = prep_for_clusterperm(df_rsa, test_model, test_orth)

    tests_dict = dict(
        single=X[..., STREAMS.index("single")],
        dual=X[..., STREAMS.index("dual")],
    )
    tests_dict["diff"] = tests_dict["single"] - tests_dict["dual"]

    distr = np.full((len(tests), niterations), np.nan)
    for itest, (test, X) in enumerate(tqdm(tests_dict.items())):

        # Generate permutation distribution
        for iteration in range(niterations):
            Xperm = perm_X_1samp(X, rng)
            tvals, pvals = scipy.stats.ttest_1samp(Xperm, **ttest_kwargs)
            clusters = return_clusters(pvals < thresh)
            distr[itest, iteration] = get_max_stat(clusters, clusterstat, tvals)

        # Get observed statistics and evaluate significance
        tvals_obs, pvals_obs = scipy.stats.ttest_1samp(X, **ttest_kwargs)
        clusters_obs = return_clusters(pvals_obs < thresh)
        (
            permdistr_dict[test_model][test]["clusterthresh_stat"],
            permdistr_dict[test_model][test]["cluster_stats"],
            permdistr_dict[test_model][test]["sig_clusters"],
            permdistr_dict[test_model][test]["pvals"],
        ) = get_significance(
            distr[itest, ...], clusterstat, clusters_obs, tvals_obs, clusterthresh
        )

    # Save permutation distributions as DataFrame
    df_distr = pd.DataFrame(distr).T
    df_distr.columns = tests
    df_distr["model"] = test_model
    dfs.append(df_distr)

df_distr = pd.concat(dfs).reset_index(drop=True)

# %%
# Save permutation results
with open(fname_perm, "w") as fout:
    json.dump(permdistr_dict, fout, indent=4, sort_keys=True)
    fout.write("\n")

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
hue = "model"
style = "stream"
if comp_anticomp_timecourse:
    hue, style = style, hue  # switch
    rsa_colors = {"single": "C6", "dual": "C7"}
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
            hue=hue,
            style=style,
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
        if bool(iax) != test_orth or comp_anticomp_timecourse:
            continue
        skipped_clus = []
        y = ax.get_ylim()[0]
        for test_model in test_models:
            c = rsa_colors[test_model]
            for test in tests:
                clusters = permdistr_dict[test_model][test]["sig_clusters"]
                if len(clusters) == 0:
                    continue
                ls = {test: _ls for test, _ls in zip(tests, ["-", "--", ":"])}[test]
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
# compare comp / acomp in window
if comp_anticomp_timecourse:
    comp_acomp_win = (0.2, 0.6)

    _df = (
        df_rsa[
            (df_rsa["orth"] == test_orth)
            & (df_rsa["time"] >= comp_acomp_win[0])
            & (df_rsa["time"] <= comp_acomp_win[1])
        ]
        .groupby(["subject", "stream", "model"])
        .mean()
        .reset_index()
    )

    with sns.plotting_context("talk"):
        fig, ax = plt.subplots()
        sns.pointplot(
            x="stream",
            order=STREAMS,
            y="similarity",
            hue="model",
            data=_df,
            ci=68,
            dodge=True,
            ax=ax,
        )
        sns.despine(fig)
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title(comp_acomp_win)

    dfs = []

    # paired ttests
    for stream in STREAMS:
        x = _df[
            (_df["stream"] == stream) & (_df["model"] == np.unique(_df["model"])[0])
        ]["similarity"]
        y = _df[
            (_df["stream"] == stream) & (_df["model"] == np.unique(_df["model"])[1])
        ]["similarity"]
        df_stats = pingouin.ttest(x, y, paired=True)
        df_stats["stream"] = stream
        dfs.append(df_stats)

    df_stats = pd.concat(dfs).reset_index(drop=True)
    df_stats

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
