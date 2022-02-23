"""Figure 4 plot - neurometrics."""
# %%
# Imports
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import ANALYSIS_DIR_LOCAL, STREAMS
from utils import prep_to_plot

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

window_sel = (0.2, 0.6)
# %%
# File paths
fname_fig3 = analysis_dir / "figures" / "fig3_pre.pdf"

fname_mean_rdm_template = str(analysis_dir / "derived_data" / "rdm_{}.npy")

fname_grids = analysis_dir / "derived_data" / "neurometrics_grids.npy"
fname_scatters = analysis_dir / "derived_data" / "neurometrics_scatters.npy"
fname_bs_ks = analysis_dir / "derived_data" / "neurometrics_bs_ks.npy"
# %%
# Start figure 3
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout()

# %%
# plot mean RDMs
# find vmin and vmax
mins, maxs = [], []
for istream, stream in enumerate(STREAMS):

    fstr = f"{stream}_{window_sel[0]}_{window_sel[1]}"
    fname = fname_mean_rdm_template.format(fstr)
    arr = np.load(fname)
    mins.append(np.nanmin(prep_to_plot(arr)))
    maxs.append(np.nanmax(prep_to_plot(arr)))
vmin = min(mins)
vmax = max(maxs)


with sns.plotting_context("talk"):
    for istream, stream in enumerate(STREAMS):

        fstr = f"{stream}_{window_sel[0]}_{window_sel[1]}"
        fname = fname_mean_rdm_template.format(fstr)
        arr = np.load(fname)
        plotrdm = prep_to_plot(arr)

        ax = axs[0, istream]

        im = ax.imshow(plotrdm, vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.125)
        if istream == 1:
            cbar = plt.colorbar(im, cax=cax, label="Mahalanobis distance")
        else:
            cax.axis("off")

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

        ax.set_title(
            stream.capitalize() + f" ({' - '.join((str(i) for i in window_sel))} s)"
        )

# %%
# Plot neurometric maps
# single grid, single mask, dual grid, dual mask
grids = np.load(fname_grids)

vmin = 0
vmax = max(grids[0, ...].max(), grids[2, ...].max())

# single xs, single ys, dual xs, dual ys
scatters = np.load(fname_scatters)

# biases, kappas
biases, kappas = np.load(fname_bs_ks)[np.array([0, 1]), ...]
idx_bias_zero = (np.abs(biases - 0.0)).argmin()
idx_kappa_one = (np.abs(kappas - 1.0)).argmin()

rotate = True
log = False
with sns.plotting_context("talk"):
    for istream, stream in enumerate(STREAMS):
        ax = axs[1, istream]

        idxs = np.array([(0, 1), (2, 3)][istream])
        grid_mean, mask = grids[idxs, ...]
        _xs, _ys = scatters[idxs, ...]
        if rotate:
            _xs, _ys = _ys, _xs

        # rotate
        origin = "upper"
        if rotate:
            grid_mean = grid_mean.T
            mask = mask.T
            origin = "lower"

        if log:
            ax.set_xscale("log")

        _ = ax.imshow(
            grid_mean,
            origin=origin,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            alpha=mask,
            cmap="magma",
            aspect="auto",
        )

        # tweak to get colorbar without alpha mask
        _, tweak_ax = plt.subplots()
        im = tweak_ax.imshow(
            grid_mean,
            origin=origin,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            aspect="auto",
        )
        plt.close(_)

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.125)
        if istream == 1:
            cbar = plt.colorbar(im, cax=cax, label="Δ Pearson's r")
            cbar_ticks = np.round(np.linspace(0, vmax, 4), 2)
            cbar.set_ticks(cbar_ticks)
            cbar.ax.set_yticklabels(["<=0"] + [f"{i}" for i in cbar_ticks[1:]])
        else:
            cax.axis("off")

        # plot subj maxima
        ax.scatter(
            _xs,
            _ys,
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
        if rotate:
            ax.axhline(idx_bias_zero, color="white", ls="--")
            ax.axvline(idx_kappa_one, color="white", ls="--")
        else:
            ax.axvline(idx_bias_zero, color="white", ls="--")
            ax.axhline(idx_kappa_one, color="white", ls="--")

        # settings
        if rotate:
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            yticklabels = (
                [""]
                + [f"{i:.2f}" for i in biases[(ax.get_yticks()[1:-1]).astype(int)]]
                + [""]
            )
            xticklabels = (
                [""]
                + [f"{i:.1f}" for i in kappas[(ax.get_xticks()[1:-1]).astype(int)]]
                + [""]
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="FixedFormatter .* FixedLocator",
                )
                ax.set(
                    title=stream.capitalize(),
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    ylabel="Bias (b)",
                    xlabel="Kappa (k)",
                )
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            xticklabels = (
                [""]
                + [f"{i:.2f}" for i in biases[(ax.get_xticks()[1:-1]).astype(int)]]
                + [""]
            )
            yticklabels = (
                [""]
                + [f"{i:.1f}" for i in kappas[(ax.get_yticks()[1:-1]).astype(int)]]
                + [""]
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="FixedFormatter .* FixedLocator",
                )
                ax.set(
                    title=stream.capitalize(),
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    xlabel="Bias (b)",
                    ylabel="Kappa (k)",
                )

# %%
# Final settings and save
fig.savefig(fname_fig3, bbox_inches="tight")

# %%
