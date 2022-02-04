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

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import statsmodels.stats.multitest
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL, NUMBERS, STREAMS, SUBJS
from utils import calc_rdm, eq1, rdm2vec

# %%
# Settings
# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

grid_res = 101
if False:
    kappas = np.linspace(0.4, 4.0, grid_res)
    biases = np.linspace(-1.0, 1.0, int(grid_res))
else:
    kappas = np.linspace(0.5, 10.0, grid_res)
    biases = np.linspace(-0.75, 0.75, int(grid_res))

bias_kappa_combis = list(itertools.product(biases, kappas))

idx_bias_zero = (np.abs(biases - 0.0)).argmin()
idx_kappa_one = (np.abs(kappas - 1.0)).argmin()

window_sel = (0.2, 0.6)  # representative window, look at RSA timecourse figure

pthresh = 0.05

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"

mahal_dir = data_dir / "derivatives" / "rsa" / "rdms_mahalanobis"

fname_rdm_template = str(mahal_dir / "sub-{:02}_stream-{}_rdm-mahal.npy")
fname_times = mahal_dir / "times.npy"

# %%
# Get times for RDM timecourses
times = np.load(fname_times)
idx_start = np.nonzero(times == window_sel[0])[0][0]
idx_stop = np.nonzero(times == window_sel[1])[0][0]

# %%
# Load all rdm_times and form mean
rdm_mean_streams_subjs = np.full(
    (len(NUMBERS), len(NUMBERS), len(STREAMS), len(SUBJS)), np.nan
)
for isub, sub in enumerate(tqdm(SUBJS)):
    for istream, stream in enumerate(STREAMS):
        rdm_times = np.load(fname_rdm_template.format(sub, stream))
        rdm_mean = np.mean(rdm_times[..., idx_start:idx_stop], axis=-1)
        rdm_mean_streams_subjs[..., istream, isub] = rdm_mean

# %%
# Calculate model RDMs
numbers_rescaled = np.interp(NUMBERS, (NUMBERS.min(), NUMBERS.max()), (-1.0, +1.0))

model_rdms = np.full((len(NUMBERS), len(NUMBERS), len(bias_kappa_combis)), np.nan)
for i, (bias, kappa) in enumerate(tqdm(bias_kappa_combis)):
    dv_vector = eq1(numbers_rescaled, bias=bias, kappa=kappa)
    dv_rdm = calc_rdm(dv_vector, normalize=True)
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
        rdm_mean_vec = rdm2vec(rdm_mean, lower_tri=True)

        for icombi, (bias, kappa) in enumerate(bias_kappa_combis):

            rdm_model = model_rdms[..., icombi]
            rdm_model_vec = rdm2vec(rdm_model, lower_tri=True)
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
max_coords_xy = np.full((2, len(STREAMS), len(SUBJS)), np.nan)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout()
for istream, stream in enumerate(STREAMS):

    ax = axs.flat[istream]

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
        grid_mean, origin="upper", interpolation="nearest", vmin=0, alpha=mask
    )

    # tweak to get colorbar without alpha mask
    _, tweak_ax = plt.subplots()
    im = tweak_ax.imshow(grid_mean, origin="upper", interpolation="nearest", vmin=0)

    # plot colorbar
    cbar = fig.colorbar(
        im, ax=ax, orientation="horizontal", label="Δ Pearson's r", shrink=0.75
    )
    cbar_ticks = np.linspace(0, im.get_array().max(), 4)
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
    ax.set(
        title=stream,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        xlabel="bias (b)",
        ylabel="kappa (k)",
    )

    title = (
        "Improved model correlation relative to linear model (b=0, k=1)\n"
        f"Transparent mask shows significant values at p={pthresh} (FDR corrected)"
    )
    fig.suptitle(title, y=1.15)
# %%
# Do the same on grand mean RDMs -- no stats possible
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.tight_layout()
grid_streams_grandmean = np.full((len(kappas), len(biases), len(STREAMS)), np.nan)
for istream, stream in enumerate(tqdm(STREAMS)):
    rdm_grandmean = np.mean(rdm_mean_streams_subjs[..., istream, :], axis=-1)
    rdm_grandmean_model_vec = rdm2vec(rdm_grandmean, lower_tri=True)

    for icombi, (bias, kappa) in enumerate(bias_kappa_combis):

        # Model correlations
        rdm_model = model_rdms[..., icombi]
        rdm_model_vec = rdm2vec(rdm_model, lower_tri=True)
        corr, _ = scipy.stats.pearsonr(rdm_grandmean_model_vec, rdm_model_vec)

        idx_bias = np.nonzero(biases == bias)[0][0]
        idx_kappa = np.nonzero(kappas == kappa)[0][0]
        grid_streams_grandmean[idx_kappa, idx_bias, istream] = corr

    # Make relative to b=0, k=1
    corr_ref = grid_streams_grandmean[idx_kappa_one, idx_bias_zero, istream]
    grid_streams_grandmean[..., istream] -= corr_ref

    # plot
    ax = axs.flat[istream]
    im = ax.imshow(
        grid_streams_grandmean[..., istream],
        origin="upper",
        interpolation="nearest",
        vmin=0,
    )
    # plot colorbar
    cbar = fig.colorbar(
        im, ax=ax, orientation="horizontal", label="Δ Pearson's r", shrink=0.75
    )
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
    ax.set(
        title=stream,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        xlabel="bias (b)",
        ylabel="kappa (k)",
    )

    title = (
        "Improved model correlation relative to linear model (b=0, k=1)\n"
        "Based on grand mean RDMs"
    )
    fig.suptitle(title, y=1.15)
# %%
