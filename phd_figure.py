"""Create a figure to visualize (anti-)compression and bias.

Required packages:
- numpy
- seaborn
- matplotlib
"""

# %%
# Imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Define our equation to obtain transformed subjective decision values


def eq1(X, bias, kappa):
    """Implement equation 1 from Spitzer et al. 2017 [1]_, [2]_.

    Parameters
    ----------
    X : np.ndarray, shape(n,)
        The input values, normalized to range [-1, 1].
    bias : float
        The bias parameter in the range [-1, 1].
    kappa : float
        The kappa parameter in the range [0, 20].

    Returns
    -------
    dv : np.ndarray, shape(n,)
        The subjective decision values, transformed from `X`

    References
    ----------
    .. [1] Spitzer, B., Waschke, L. & Summerfield, C. Selective overweighting of larger
           magnitudes during noisy numerical comparison. Nat Hum Behav 1, 0145 (2017).
           https://doi.org/10.1038/s41562-017-0145
    .. [2] https://github.com/summerfieldlab/spitzer_etal_2017/blob/master/psymodfun.m
    """
    dv = np.sign(X + bias) * (np.abs(X + bias) ** kappa)
    return dv


# %%
# Create a figure
n = 900
X = np.linspace(-1, 1, n)
xs = np.linspace(1, 9, n)
kappas = dict(zip(["compressive\n(k<1)", "anti-\ncompressive\n(k>1)"], [0.5, 3]))


with sns.plotting_context("talk", font_scale=1):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    axcount = 0
    for title, kappa in kappas.items():
        ax = axs.flat[axcount]
        ax.plot(xs, eq1(X, bias=0, kappa=kappa), label=f"k={kappa}")
        vals = eq1(X, bias=0, kappa=1)
        ax.axline((xs[0], vals[0]), (xs[1], vals[1]), c="black", lw=0.5, ls="--")
        axcount += 1
        ax.text(x=2.5, y=0.5, s=title, ha="center", va="center")

    axs.flat[0].legend(frameon=False, loc="lower right")
    axs.flat[1].legend(frameon=False, loc="lower right")

    for ax in axs:
        ax.set_xticks(np.arange(1, 10))
        ax.set_yticks(np.linspace(-1, 1, 3))
        ax.set(ylabel="Subjective decision value", xlabel="Sample value")
        ax.axhline(0, c="black", lw=0.5, ls="--")
        ax.axvline(5, c="black", lw=0.5, ls="--")

    axs.flat[0].set_xlabel("")

    fig.tight_layout()
    sns.despine(fig)

# save
fname = "phd_figure.png"
fig.savefig(fname, bbox_inches="tight", dpi=600)

# %%
