"""Calculate RSA neurometrics.

- import subj/stream wise rdm_times arrays
- for each subj/stream, average over time window: 9x9 RDM
- create an array of numberline RDMs with different parameters each: kappa, bias
- for each subj/stream/meantime RDM, correlate with all model RDMs --> grid
- plot mean over grids for each stream
- plot individual grid maxima
- plot mean grid maximum
- make maps relative to "linear" map: all models that lead to worse correlation
  are coded "<= 0" (minimum color)
- calculate t values over participant correlations: ttest or wilcoxon

"""
# %%
# Imports
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL, STREAMS, SUBJS
from model_rdms import get_models_dict

# %%
# Settings
# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

grid_res = 101
opt = 2
if opt == 0:
    kappas = np.linspace(0.4, 4.0, grid_res)
    biases = np.linspace(-1.0, 1.0, int(grid_res))
elif opt == 1:
    kappas = np.linspace(0.5, 10.0, grid_res)
    biases = np.linspace(-0.75, 0.75, int(grid_res))
elif opt == 2:
    kappas = np.linspace(0.5, 10.0, grid_res)
    biases = np.linspace(-0.5, 0.5, int(grid_res))
else:
    raise RuntimeError(f"unknown 'opt': {opt}")

bias_kappa_combis = list(itertools.product(biases, kappas))

idx_bias_zero = (np.abs(biases - 0.0)).argmin()
idx_kappa_one = (np.abs(kappas - 1.0)).argmin()

window_sel = (0.2, 0.6)  # representative window, look at RSA timecourse figure

pthresh = 0.05

subtract_maps = True

rdm_size = "18x18"
ndim = int(rdm_size.split("x")[0])

# "Pearson' r", "Kendall's tau-b", "Spearman's rho"
corr_method = "Pearson's r"

# whether or not to orthogonalize the model RDMs
orth = True

# which models to orthogonalize "numberline" with?
modelnames = ["digit", "color", "numberline"]

if rdm_size == "9x9":
    modelnames = ["numberline"]
    orth = False
    print("For 9x9 neurometrics, always run without orth and only numberline.")

# overwrite for saving?
overwrite = False

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"

mahal_dir = data_dir / "derivatives" / "rsa" / rdm_size / "rdms_mahalanobis"

fname_rdm_template = str(mahal_dir / "sub-{:02}_stream-{}_rdm-mahal.npy")
fname_times = mahal_dir / "times.npy"

fname_params = analysis_dir / "derived_data" / "neurometrics_params.tsv"

# %%
# Get times for RDM timecourses
times = np.load(fname_times)
idx_start = np.nonzero(times == window_sel[0])[0][0]
idx_stop = np.nonzero(times == window_sel[1])[0][0]

# %%
# Load all rdm_times and form mean
rdm_mean_streams_subjs = np.full((ndim, ndim, len(STREAMS), len(SUBJS)), np.nan)
for isub, sub in enumerate(tqdm(SUBJS)):
    for istream, stream in enumerate(STREAMS):
        rdm_times = np.load(fname_rdm_template.format(sub, stream))
        rdm_mean = np.mean(rdm_times[..., idx_start:idx_stop], axis=-1)
        rdm_mean_streams_subjs[..., istream, isub] = rdm_mean

# %%
# Calculate model RDMs
key = "orth" if orth else "no_orth"
model_rdms = np.full((ndim, ndim, len(bias_kappa_combis)), np.nan)
for i, (bias, kappa) in enumerate(tqdm(bias_kappa_combis)):
    models_dict = get_models_dict(rdm_size, modelnames, orth, bias=bias, kappa=kappa)
    dv_rdm = models_dict[key]["numberline"]
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
        rdm_mean = rdm_mean_streams_subjs[..., istream, isub]
        rdm_mean_vec = squareform(rdm_mean)

        for icombi, (bias, kappa) in enumerate(bias_kappa_combis):

            rdm_model = model_rdms[..., icombi]
            rdm_model_vec = squareform(rdm_model)
            if corr_method == "Pearson's r":
                corr, _ = scipy.stats.pearsonr(rdm_mean_vec, rdm_model_vec)
            elif corr_method == "Kendall's tau-b":
                corr, _ = scipy.stats.kendalltau(rdm_mean_vec, rdm_model_vec)
            else:
                assert corr_method == "Spearman's rho"
                corr, _ = scipy.stats.spearmanr(rdm_mean_vec, rdm_model_vec)

            idx_bias = np.nonzero(biases == bias)[0][0]
            idx_kappa = np.nonzero(kappas == kappa)[0][0]
            grid_streams_subjs[idx_kappa, idx_bias, istream, isub] = corr

# %%
# Normalize maps to be relative to bias=0, kappa=1
if subtract_maps:
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
max_coords_xy = np.full((2, len(STREAMS), len(SUBJS)), np.nan)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout()
for istream, stream in enumerate(STREAMS):

    ax = axs.flat[istream]

    # settings
    cbarlabel = corr_method
    vmin = None
    vmax = max(
        grid_streams_subjs[..., 0, :].mean(axis=-1).max(),
        grid_streams_subjs[..., 1, :].mean(axis=-1).max(),
    )
    if subtract_maps:
        cbarlabel = "Δ " + cbarlabel
        vmin = 0

    # Calculate subj wise maxima
    for isub, sub in enumerate(SUBJS):
        shape = grid_streams_subjs[..., istream, isub].shape
        argmax = np.argmax(grid_streams_subjs[..., istream, isub])
        max_coords_xy[..., istream, isub] = np.unravel_index(argmax, shape)[::-1]

    # plot mean grid
    grid_mean = np.mean(grid_streams_subjs[..., istream, :], axis=-1)
    mask = sig_masks_streams[..., istream]
    mask[grid_mean <= 0] = alpha_val_mask
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
    if subtract_maps:
        uptick = (
            max(im.get_array().max(), vmax)
            if vmax is not None
            else im.get_array().max()
        )
        cbar_ticks = np.linspace(0, uptick, 4)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_xticklabels(["<=0"] + [f"{i:.2}" for i in cbar_ticks[1:]])

    # plot subj maxima
    ax.scatter(
        max_coords_xy[..., istream, :][0],
        max_coords_xy[..., istream, :][1],
        color="red",
        s=4,
        zorder=10,
    )

    # plot mean maximum
    mean_max_xy = np.unravel_index(np.argmax(grid_mean), grid_mean.shape)[::-1]

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
    if subtract_maps:
        title = (
            "Improved model correlation relative to linear model (b=0, k=1)\n" + title
        )
    title = f"rdm_size={rdm_size}, orth={orth}\n" + title

    fig.suptitle(title, y=1.15)

# %%
# Plot single subj maps
for stream in STREAMS:
    istream = STREAMS.index(stream)
    fig, axs = plt.subplots(5, 6, figsize=(10, 10))
    for isub, sub in enumerate(SUBJS):
        grid = grid_streams_subjs[..., istream, isub]
        ax = axs.flat[isub]
        ax.imshow(grid)

        ax.scatter(
            max_coords_xy[..., istream, isub][0],
            max_coords_xy[..., istream, isub][1],
            color="red",
            s=4,
            zorder=10,
        )

        ax.set_axis_off()
    fig.suptitle(f"Single subjects: {stream}")

# %%
# Save single subj bias and kappa maxima
dfs = []
for stream in STREAMS:
    istream = STREAMS.index(stream)

    bs = biases[max_coords_xy[0, istream, :].astype(int)]
    ks = kappas[max_coords_xy[1, istream, :].astype(int)]
    df = pd.DataFrame([bs, ks]).T
    df.columns = ["bias", "kappa"]
    df.insert(0, "stream", stream)
    df.insert(0, "subject", SUBJS)
    dfs.append(df)

df = pd.concat(dfs).sort_values(["subject", "stream"]).reset_index(drop=True)
df["rdm_size"] = rdm_size
df["subtract_maps"] = subtract_maps
df["orth"] = orth
df["corr_method"] = corr_method

if fname_params.exists() and not overwrite:
    print("Params file exists and overwrite is set to False, not saving anything.")
else:
    df.to_csv(fname_params, sep="\t", na_rep="n/a", index=False)

# %%
# Plot maps based on grand mean RDMs -- no stats possible
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout()
grid_streams_grandmean = np.full((len(kappas), len(biases), len(STREAMS)), np.nan)
for istream, stream in enumerate(tqdm(STREAMS)):
    rdm_grandmean = np.mean(rdm_mean_streams_subjs[..., istream, :], axis=-1)
    rdm_grandmean_model_vec = squareform(rdm_grandmean)

    for icombi, (bias, kappa) in enumerate(bias_kappa_combis):

        # Model correlations
        rdm_model = model_rdms[..., icombi]
        rdm_model_vec = squareform(rdm_model)
        if corr_method == "Pearson's r":
            corr, _ = scipy.stats.pearsonr(rdm_grandmean_model_vec, rdm_model_vec)
        elif corr_method == "Kendall's tau-b":
            corr, _ = scipy.stats.kendalltau(rdm_grandmean_model_vec, rdm_model_vec)
        else:
            assert corr_method == "Spearman's rho"
            corr, _ = scipy.stats.spearmanr(rdm_grandmean_model_vec, rdm_model_vec)
        idx_bias = np.nonzero(biases == bias)[0][0]
        idx_kappa = np.nonzero(kappas == kappa)[0][0]
        grid_streams_grandmean[idx_kappa, idx_bias, istream] = corr

    # settings
    cbarlabel = corr_method
    vmin = None
    if subtract_maps:
        cbarlabel = "Δ " + cbarlabel
        vmin = 0

    # Make relative to b=0, k=1
    if subtract_maps:
        corr_ref = grid_streams_grandmean[idx_kappa_one, idx_bias_zero, istream]
        grid_streams_grandmean[..., istream] -= corr_ref

    # plot
    ax = axs.flat[istream]
    im = ax.imshow(
        grid_streams_grandmean[..., istream],
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
    )
    # plot colorbar
    cbar = fig.colorbar(
        im, ax=ax, orientation="horizontal", label=cbarlabel, shrink=0.75
    )
    if subtract_maps:
        cbar_ticks = np.linspace(0, im.get_array().max(), 4)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_xticklabels(["<=0"] + [f"{i:.2}" for i in cbar_ticks[1:]])

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

    title = "Based on grand mean RDMs"
    if subtract_maps:
        title = (
            "Improved model correlation relative to linear model (b=0, k=1)\n" + title
        )
    title = f"rdm_size={rdm_size}, orth={orth}\n" + title

    fig.suptitle(title, y=1.15)
# %%
