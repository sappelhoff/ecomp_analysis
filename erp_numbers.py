"""Compute and plot ERPs over subjects."""
# %%
# Imports
import itertools
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pingouin
import scipy.stats
import seaborn as sns
import statsmodels.stats.multitest
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from config import (
    ANALYSIS_DIR_LOCAL,
    DATA_DIR_EXTERNAL,
    NUMBERS,
    P3_GROUP_CERCOR,
    P3_GROUP_NHB,
    STREAMS,
    SUBJS,
)
from model_rdms import get_models_dict
from utils import calc_rdm, eq1

# %%
# Settings
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

overwrite = False

subjects = SUBJS
baseline = (None, 0)

mean_times = (0.3, 0.7)

p3_group = [P3_GROUP_CERCOR, P3_GROUP_NHB][1]  # 0 or 1

numbers_rescaled = np.interp(NUMBERS, (NUMBERS.min(), NUMBERS.max()), (-1, +1))

# %%
# Prepare file paths

derivatives = data_dir / "derivatives"

erp_dir = derivatives / "erps" / f"{baseline}-{mean_times}"
erp_dir.mkdir(exist_ok=True, parents=True)
fname_template_erp = "sub-{:02}_stream-{}_number-{}_ave.fif.gz"

fname_erps = analysis_dir / "derived_data" / "erps.tsv"
fname_amps = analysis_dir / "derived_data" / "erp_amps.tsv"
fname_adm = analysis_dir / "derived_data" / "erp_adm.tsv"


# %%
# Helper functions to get data
def _get_mean_amps(dict_streams, mean_times, p3_group):
    dict_mean_amps = {j: {str(i): [] for i in NUMBERS} for j in STREAMS}
    for stream, dict_numbers in dict_streams.items():
        for number, evoked_list in dict_numbers.items():
            for i, evoked in enumerate(evoked_list):
                start, stop = evoked.time_as_index(mean_times, use_rounding=True)
                # extract mean amplitude and save
                with mne.utils.use_log_level(False):
                    df_epo = (
                        evoked.to_data_frame(picks=p3_group, long_format=True)
                        .groupby("time")
                        .mean()
                        .reset_index(drop=False)
                    )

                assert df_epo.iloc[start]["time"] / 1000 == mean_times[0]
                assert df_epo.iloc[stop]["time"] / 1000 == mean_times[1]
                mean_amp = df_epo.iloc[start : stop + 1]["value"].mean()
                dict_mean_amps[stream][number] += [mean_amp]

    return dict_mean_amps


def _get_erps(sub, baseline):
    # Read data
    fname_epo = derivatives / f"sub-{sub:02}" / f"sub-{sub:02}_numbers_epo.fif.gz"
    epochs = mne.read_epochs(fname_epo, preload=True, verbose=False)

    # apply baseline
    epochs.apply_baseline(baseline)

    # Calculate cocktail blanks per stream
    cocktail_blank = {stream: epochs[stream].average().get_data() for stream in STREAMS}

    # Get cocktailblank corrected ERPs for each stream/number
    # also extract mean amplitudes
    dict_streams = {j: {str(i): [] for i in NUMBERS} for j in STREAMS}
    for stream, dict_numbers in dict_streams.items():
        for number in dict_numbers:

            # form ERP and cocktailblank correct
            sel = f"{stream}/{number}"
            evoked = epochs[sel].average()
            evoked.data = evoked.data - cocktail_blank[stream]

            # add to evokeds
            dict_numbers[number] += [evoked]

    return dict_streams


# %%
# Get subject data
dict_streams = {j: {str(i): [] for i in NUMBERS} for j in STREAMS}
if erp_dir.exists():
    # Try load ERPs from disk directly (faster)
    for stream, dict_numbers in tqdm(dict_streams.items()):
        for number in dict_numbers:
            for sub in SUBJS:
                fname = erp_dir / fname_template_erp.format(sub, stream, number)
                evokeds = mne.read_evokeds(fname, verbose=False)
                assert len(evokeds) == 1
                dict_streams[stream][number] += evokeds
else:
    # Prepare from epochs (slower)
    sub_dicts = {}
    for sub in tqdm(SUBJS):
        sub_dicts[sub] = _get_erps(sub, baseline)

    # Merge subject dicts into one dict of list
    for sub, _dstream in sub_dicts.items():
        for stream, dict_numbers in dict_streams.items():
            for number in dict_numbers:
                dict_streams[stream][number] += _dstream[stream][number]

# Get mean amps
dict_mean_amps = _get_mean_amps(dict_streams, mean_times, p3_group)

# %%
# Plot
cmap = sns.color_palette("crest_r", as_cmap=True)
for stream in STREAMS:
    fig, ax = plt.subplots()
    ax.axvspan(*mean_times, color="black", alpha=0.1)
    with mne.utils.use_log_level(False):
        mne.viz.plot_compare_evokeds(
            dict_streams[stream],
            picks=p3_group,
            combine="mean",
            show_sensors=True,
            cmap=cmap,
            ci=0.68,
            axes=ax,
            title=stream,
        )

# %%
# Save plot data as DF
df_erps = []
for stream in STREAMS:
    for number in NUMBERS:
        for isub, subj in enumerate(SUBJS):
            _ = (
                dict_streams[stream][f"{number}"][isub]
                .to_data_frame(picks=p3_group, long_format=True, time_format=None)
                .groupby("time")
                .mean()
                .reset_index()
            )
            _.insert(0, "number", number)
            _.insert(0, "stream", stream)
            _.insert(0, "subject", subj)
            _["baseline"] = [baseline] * len(_)
            _["p3_group"] = [p3_group] * len(_)
            df_erps.append(_)
df_erps = pd.concat(df_erps).reset_index(drop=True)
assert len(df_erps) == len(SUBJS) * len(STREAMS) * len(NUMBERS) * 251  # 251 timepoints
df_erps.to_csv(fname_erps, sep="\t", na_rep="n/a", index=False)
# %%
# Gather mean amplitude data into DataFrame
all_dfs = []
for stream in STREAMS:
    df_mean_amps = pd.DataFrame.from_dict(dict_mean_amps[stream])
    assert df_mean_amps.shape[0] == len(subjects)
    df_mean_amps["subject"] = subjects
    df_mean_amps = df_mean_amps.melt(id_vars=["subject"]).sort_values(
        by=["subject", "variable"]
    )
    df_mean_amps = df_mean_amps.rename(
        columns=dict(variable="number", value="mean_amp")
    )
    df_mean_amps.insert(1, "stream", stream)
    all_dfs.append(df_mean_amps)

df_mean_amps = pd.concat(all_dfs).reset_index(drop=True)
assert df_mean_amps.shape[0] == len(subjects) * len(NUMBERS) * len(STREAMS)

df_mean_amps["baseline"] = [baseline] * len(df_mean_amps)
df_mean_amps["mean_times"] = [mean_times] * len(df_mean_amps)
df_mean_amps["p3_group"] = [p3_group] * len(df_mean_amps)

# %%
# Save mean amps
df_mean_amps.to_csv(fname_amps, sep="\t", na_rep="n/a", index=False)

# %%
# Plot mean amps
fig, ax = plt.subplots()
sns.pointplot(x="number", y="mean_amp", hue="stream", data=df_mean_amps, ci=68, ax=ax)
sns.despine(fig)

# %%
# Save ERPs
for stream, dict_numbers in dict_streams.items():
    for number, evoked_list in dict_numbers.items():
        for i, evoked in enumerate(evoked_list):
            sub = SUBJS[i]
            fname = erp_dir / fname_template_erp.format(sub, stream, number)
            if fname.exists() and not overwrite:
                continue
            evoked.save(fname, overwrite)

# %%
# Repeated measures anova
stats_rm = pingouin.rm_anova(
    data=df_mean_amps, dv="mean_amp", within=["stream", "number"], subject="subject"
)
stats_rm.round(3)

# %%
# Quantify (anti-)compression
# NOTE: take abs() of dv
cmap = sns.color_palette("crest_r", as_cmap=True)

bs = np.linspace(-1, 1, 7)
ks = np.array([0, 0.1])


fig, axs = plt.subplots(1, len(bs), figsize=(10, 5), sharex=True, sharey=True)
for i, b in enumerate(bs):
    ax = axs.flat[i]
    ax.plot(
        np.linspace(-1, 1, 9),
        np.abs(eq1(numbers_rescaled, bias=0, kappa=1)),
        color="k",
        marker="o",
        label="b=0, k=1",
    )
    ax.set(title=f"bias={b:.2}", ylabel="decision weight", xlabel="X")

    for k in ks:
        ax.plot(
            np.linspace(-1, 1, 1001),
            np.abs(eq1(np.linspace(-1, 1, 1001), b, k)),
            color=cmap(k / ks.max()),
            label=f"k={k:.2f}",
        )

    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(0, color="k", linestyle="--")
    if i == 0:
        ax.legend()

fig.tight_layout()

# %%
# Neurometrics on ERP
# We assume same mental transformation as in standard model, but
# we look for the results in a measure that can only reflect "absolute values" (the ERP)

# For observed RDMs formed from the ERPs, normalizing amplitude values to [0, 1]
ndim = len(NUMBERS)
rdms_obs = np.full((ndim, ndim, len(STREAMS), len(SUBJS)), np.nan)
for meta, grp in df_mean_amps.groupby(["subject", "stream"]):
    subj, stream = meta
    isub = SUBJS.tolist().index(subj)
    istream = STREAMS.index(stream)
    dv_obs = grp.sort_values("number")["mean_amp"].to_numpy()
    rdm_obs = calc_rdm(dv_obs, normalize=True)
    rdms_obs[..., istream, isub] = rdm_obs

assert not np.isnan(rdms_obs).any()
# %%
# Calculate model RDMs
rdm_size = "9x9"
modelnames = ["numberline"]
grid_res = 101
opt = 0
if opt == 0:
    kappas = np.linspace(0.1, 6.5, grid_res)
    biases = np.linspace(-0.5, 0.5, grid_res)
else:
    raise RuntimeError(f"unknown 'opt': {opt}")
bias_kappa_combis = list(itertools.product(biases, kappas))
idx_bias_zero = (np.abs(biases - 0.0)).argmin()
idx_kappa_one = (np.abs(kappas - 1.0)).argmin()

pthresh = 0.05

model_rdms = np.full((ndim, ndim, len(bias_kappa_combis)), np.nan)
for i, (bias, kappa) in enumerate(tqdm(bias_kappa_combis)):
    models_dict = get_models_dict(
        rdm_size, modelnames, orth=False, bias=bias, kappa=kappa, abs_dv=True
    )
    dv_rdm = models_dict["no_orth"]["numberline"]
    model_rdms[..., i] = dv_rdm

assert not np.isnan(model_rdms).any()

# %%
# Correlate ERP-RDMs and models --> one grid per subj/stream
grid_streams_subjs = np.full(
    (len(kappas), len(biases), len(STREAMS), len(SUBJS)), np.nan
)
for isub, sub in enumerate(tqdm(SUBJS)):
    for istream, stream in enumerate(STREAMS):

        # Get ERP ERM
        rdm_mean = rdms_obs[..., istream, isub]
        rdm_mean_vec = squareform(rdm_mean)

        for icombi, (bias, kappa) in enumerate(bias_kappa_combis):

            rdm_model = model_rdms[..., icombi]
            rdm_model_vec = squareform(rdm_model)
            corr, _ = scipy.stats.pearsonr(rdm_mean_vec, rdm_model_vec)
            idx_bias = np.nonzero(biases == bias)[0][0]
            idx_kappa = np.nonzero(kappas == kappa)[0][0]
            grid_streams_subjs[idx_kappa, idx_bias, istream, isub] = corr

# %%
# Normalize maps to be relative to bias=0, kappa=1
rng = np.random.default_rng(42)
for isub, sub in enumerate(tqdm(SUBJS)):
    for istream, stream in enumerate(STREAMS):
        corr_ref = grid_streams_subjs[idx_kappa_one, idx_bias_zero, istream, isub]
        grid_streams_subjs[..., istream, isub] -= corr_ref

        # don't make the b=0, k=1 cell zero for all subjs. Add tiny amount
        # of random noise, so that down-the-line tests don't run into NaN problems
        noise = rng.normal() * 1e-5
        grid_streams_subjs[idx_kappa_one, idx_bias_zero, istream, isub] += noise

# %%
# Calculate 1 samp t-tests against 0 for each cell to test significance
pval_maps_streams = np.full((len(kappas), len(biases), len(STREAMS)), np.nan)
for istream, stream in enumerate(STREAMS):
    data = grid_streams_subjs[..., istream, :]
    _, pvals = scipy.stats.ttest_1samp(a=data, popmean=0, axis=-1, nan_policy="raise")
    pval_maps_streams[..., istream] = pvals

# %%
# Create a mask for plotting the significant values in grid
# use B/H FDR correction
alpha_val_mask = 0.75
sig_masks_streams = np.full_like(pval_maps_streams, np.nan)
for istream, stream in enumerate(STREAMS):
    pvals = pval_maps_streams[..., istream]
    sig, _ = statsmodels.stats.multitest.fdrcorrection(pvals.flatten(), alpha=pthresh)
    sig = sig.reshape(pvals.shape)

    # all non-significant values have a lower "alpha value" in the plot
    mask_alpha_vals = sig.copy().astype(float)
    mask_alpha_vals[mask_alpha_vals == 0] = alpha_val_mask
    sig_masks_streams[..., istream] = mask_alpha_vals

    # NOTE: Need to lower alpha values of cells with corr <= 0
    #       that are still significant before plotting
    #       E.g., a cell significantly worsens correlation -> adjust alpha

# %%
# Plot grid per stream
mean_max_xys = []
grids = []
scatters = []
max_coords_xy = np.full((2, len(STREAMS), len(SUBJS)), np.nan)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout()
for istream, stream in enumerate(STREAMS):

    ax = axs.flat[istream]

    # settings
    cbarlabel = "Δ Pearson's r"
    vmin = 0
    vmax = max(
        grid_streams_subjs[..., 0, :].mean(axis=-1).max(),
        grid_streams_subjs[..., 1, :].mean(axis=-1).max(),
    )

    # Calculate subj wise maxima
    for isub, sub in enumerate(SUBJS):
        shape = grid_streams_subjs[..., istream, isub].shape
        argmax = np.argmax(grid_streams_subjs[..., istream, isub])
        max_coords_xy[..., istream, isub] = np.unravel_index(argmax, shape)[::-1]

    # plot mean grid
    grid_mean = np.mean(grid_streams_subjs[..., istream, :], axis=-1)
    mask = sig_masks_streams[..., istream]
    mask[grid_mean <= 0] = alpha_val_mask
    grids.append((grid_mean, mask))
    _ = ax.imshow(
        grid_mean,
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        alpha=mask,
    )

    # tweak to get colorbar without alpha mask
    _, tweak_ax = plt.subplots()
    im = tweak_ax.imshow(
        grid_mean, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax
    )
    plt.close(_)

    # plot colorbar
    cbar = fig.colorbar(
        im, ax=ax, orientation="horizontal", label=cbarlabel, shrink=0.75
    )

    uptick = (
        max(im.get_array().max(), vmax) if vmax is not None else im.get_array().max()
    )
    cbar_ticks = np.linspace(0, uptick, 4)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_xticklabels(["<=0"] + [f"{i:.2}" for i in cbar_ticks[1:]])

    # plot subj maxima
    _xs = max_coords_xy[..., istream, :][0]
    _ys = max_coords_xy[..., istream, :][1]
    scatters.append((_xs, _ys))
    ax.scatter(
        _xs,
        _ys,
        color="red",
        s=4,
        zorder=10,
    )

    # plot mean maximum
    mean_max_xy = np.unravel_index(np.argmax(grid_mean), grid_mean.shape)[::-1]
    mean_max_xys += [mean_max_xy]
    ax.scatter(
        mean_max_xy[0],
        mean_max_xy[1],
        color="red",
        s=24,
        marker="d",
        zorder=10,
    )

    # lines
    ax.axvline(idx_bias_zero, color="white", ls="--")
    ax.axhline(idx_kappa_one, color="white", ls="--")

    # settings
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    xticklabels = (
        [""] + [f"{i:.2f}" for i in biases[(ax.get_xticks()[1:-1]).astype(int)]] + [""]
    )
    yticklabels = (
        [""] + [f"{i:.1f}" for i in kappas[(ax.get_yticks()[1:-1]).astype(int)]] + [""]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="FixedFormatter .* FixedLocator"
        )
        ax.set(
            title=stream,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            xlabel="bias (b)",
            ylabel="kappa (k)",
        )

    title = f"Transparent mask shows significant values at p={pthresh} (FDR corrected)"
    title = "Improved model correlation relative to linear model (b=0, k=1)\n" + title
    title = f"rdm_size={rdm_size}, orth=False\n" + title

    fig.suptitle(title, y=1.15)

# %%
# plot model curves - ADM: Amplitude Difference Matrix
df_adm = []
for isub, subj in enumerate(SUBJS):
    for istream, stream in enumerate(STREAMS):
        d = dict(subject=subj, stream=stream)
        x, y = max_coords_xy[..., istream, isub]
        d["bias"] = biases[int(x)]
        d["kappa"] = kappas[int(y)]
        d["number"] = NUMBERS
        d["dv"] = eq1(numbers_rescaled, bias=d["bias"], kappa=d["kappa"])
        d["dv_abs"] = np.abs(d["dv"])
        d["dv_abs_k1"] = np.abs(eq1(numbers_rescaled, bias=d["bias"], kappa=1))

        _ = pd.DataFrame.from_dict(d)
        df_adm.append(_)

df_adm = pd.concat(df_adm).reset_index(drop=True)

plot_single_sub = False
plot_map_maxima = False
plot_k1 = False
plot_b0_k1 = True
with sns.plotting_context("talk"):
    fig, ax = plt.subplots()
    data = df_adm
    sns.pointplot(
        x="number", y="dv_abs", hue="stream", data=data, ci=68, ax=ax, dodge=False
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        frameon=False,
        title=None,
        handles=handles,
        labels=[i.capitalize() for i in labels],
    )
    ax.set(xlabel="", ylabel="Absolute decision weight")
    sns.despine(ax=ax)

    # plot single subjs
    if plot_single_sub:
        for subj in SUBJS:
            for istream, stream in enumerate(STREAMS):
                data = df_adm[
                    (df_adm["subject"] == subj) & (df_adm["stream"] == stream)
                ]
                c = f"C{istream}"
                ax.plot(
                    data["number"] - 1,
                    data["dv_abs"],
                    color=c,
                    alpha=0.5,
                    lw=0.5,
                    zorder=0,
                )

    # plot "map maxima"
    if plot_map_maxima:
        for istream, stream in enumerate(STREAMS):
            x, y = mean_max_xys[istream]
            vals = np.abs(eq1(numbers_rescaled, bias=biases[x], kappa=kappas[y]))
            ax.plot(NUMBERS - 1, vals, color=f"C{istream}", ls="--")

    # plot with k=1
    if plot_k1:
        for stream, grp in data.groupby("stream"):
            vals = grp.groupby("number")["dv_abs_k1"].mean().to_numpy()
            ax.plot(
                NUMBERS - 1, vals, color=f"C{STREAMS.index(stream)}", ls="--", zorder=0
            )

    # plot with mean bias, k=1
    if plot_b0_k1:
        vals = np.abs(eq1(numbers_rescaled, bias=df_adm["bias"].mean(), kappa=1))
        ax.plot(NUMBERS - 1, vals, color="black", ls="--", zorder=10)

    if plot_single_sub:
        ax.set_ylim((-0.1, 3.5))

df_adm.groupby("stream")[["bias", "kappa"]].describe()

# Save ADM df
df_adm.to_csv(fname_adm, sep="\t", na_rep="n/a", index=False)

# %%
# Plot map maxima standalone
fig, ax = plt.subplots()
valss = []
for istream, stream in enumerate(STREAMS):
    x, y = mean_max_xys[istream]
    vals = np.abs(eq1(np.linspace(-1, 1, 9), bias=biases[x], kappa=kappas[y]))
    valss += [vals]
    ax.plot(np.arange(9), vals)

# %%
