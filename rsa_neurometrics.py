"""Calculate RSA neurometrics.

- import subj/stream wise rdm_times arrays
- for each subj/stream, average over time window: 9x9 RDM
- create an array of numberline RDMs with different parameters each: kappa, bias
- for each subj/stream/meantime RDM, correlate with all model RDMs --> grid
- plot mean over grids for each stream
- plot individual grid maxima
- plot mean grid maximum

"""
# %%
# Imports
import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL, SUBJS
from utils import calc_rdm, eq1, rdm2vec

# %%
# Settings
# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

numbers = np.arange(1, 10, dtype=int)
streams = ["single", "dual"]

grid_res = 101
kappas = np.linspace(0.4, 4, grid_res)
biases = np.linspace(-1, 1, int(grid_res))
bias_kappa_combis = list(itertools.product(biases, kappas))

window_sel = (0.2, 0.6)  # representative window, look at figure

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
    (len(numbers), len(numbers), len(streams), len(SUBJS)), np.nan
)
for isub, sub in enumerate(tqdm(SUBJS)):
    for istream, stream in enumerate(streams):
        rdm_times = np.load(fname_rdm_template.format(sub, stream))
        rdm_mean = np.mean(rdm_times[..., idx_start:idx_stop], axis=-1)
        rdm_mean_streams_subjs[..., istream, isub] = rdm_mean

# %%
# Calculate model RDMs
numbers_rescaled = np.interp(numbers, (numbers.min(), numbers.max()), (-1.0, +1.0))

model_rdms = np.full((len(numbers), len(numbers), len(bias_kappa_combis)), np.nan)
for i, (bias, kappa) in enumerate(tqdm(bias_kappa_combis)):
    dv_vector = eq1(numbers_rescaled, bias=bias, kappa=kappa)
    dv_rdm = calc_rdm(dv_vector, normalize=True)
    model_rdms[..., i] = dv_rdm

assert not np.isnan(model_rdms).any()

# %%
# Correlate ERP-RDMs and models --> one grid per subj/stream
grid_streams_subjs = np.full(
    (len(kappas), len(biases), len(streams), len(SUBJS)), np.nan
)
for isub, sub in enumerate(tqdm(SUBJS)):
    for istream, stream in enumerate(streams):

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
# Plot grid per stream
max_coords_xy = np.full((2, len(streams), len(SUBJS)), np.nan)
fig, axs = plt.subplots(1, 2)
fig.tight_layout()
for istream, stream in enumerate(streams):

    ax = axs.flat[istream]

    # Calculate subj wise maxima
    for isub, sub in enumerate(SUBJS):
        shape = grid_streams_subjs[..., istream, isub].shape
        argmax = np.argmax(grid_streams_subjs[..., istream, isub])
        max_coords_xy[..., istream, isub] = np.unravel_index(argmax, shape)

    # plot mean grid
    grid_mean = np.mean(grid_streams_subjs[..., istream, :], axis=-1)
    ax.imshow(grid_mean, origin="upper", interpolation="nearest")

    # plot subj maxima
    ax.scatter(
        max_coords_xy[..., istream, :][0],
        max_coords_xy[..., istream, :][1],
        color="red",
        s=4,
        zorder=10,
    )

    # plot mean maxima
    ax.scatter(
        np.mean(max_coords_xy[..., istream, :], axis=-1)[0],
        np.mean(max_coords_xy[..., istream, :], axis=-1)[1],
        color="red",
        s=24,
        marker="d",
        zorder=10,
    )

    # lines
    idx_bias_zero = (np.abs(biases - 0.0)).argmin()
    idx_kappa_one = (np.abs(kappas - 1.0)).argmin()
    ax.axvline(idx_bias_zero, color="white", ls="--")
    ax.axhline(idx_kappa_one, color="white", ls="--")

    # settings
    xticklabels = (
        [""] + [str(i) for i in biases[(ax.get_xticks()[1:-1]).astype(int)]] + [""]
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

# %%
