"""Analyze alpha freqs.

1. Load TFRs
2. plot topomaps: 2xN --> row1: single, row2: dual
3. extract mean from elecs identified in maps --> plot single vs dual

"""
# %%
# Imports
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns

from config import DATA_DIR_EXTERNAL, STREAMS, SUBJS

# %%
# Settings
data_dir = DATA_DIR_EXTERNAL
sub = 1
stream = "single"
overwrite = False

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"


# %%
# Read TFRs

singles = []
duals = []
for stream in STREAMS:
    for sub in SUBJS:
        fname_tfr = (
            derivatives / f"sub-{sub:02}" / f"sub-{sub:02}_stream-{stream}_tfr.h5"
        )
        tfr = mne.time_frequency.read_tfrs(fname_tfr)
        assert len(tfr) == 1
        tfr = tfr[0]

        if stream == "single":
            singles.append(tfr)
        else:
            duals.append(tfr)


# %%
# average over freq bands
# subj x channels x times (all "alpha power" ~8-15Hz)
single_data = np.stack([np.mean(tfr.data.copy(), 1) for tfr in singles])
dual_data = np.stack([np.mean(tfr.data.copy(), 1) for tfr in duals])


# %%
tfr.data = np.mean(np.stack([tfr.data.copy() for tfr in duals]), 0)
_ = tfr.plot(picks=["Oz"])

# %%
# Create "fake evoked" for topo plotting
info = mne.create_info(tfr.ch_names, sfreq=tfr.info["sfreq"], ch_types="eeg")
info.set_montage("easycap-M1")

single_evoked = mne.EvokedArray(
    np.mean(single_data, 0),
    info,
    tmin=-0.5,
    kind="average",
    nave=single_data.shape[0],
    comment="averaged TFRs",
)
dual_evoked = mne.EvokedArray(
    np.mean(dual_data, 0),
    info,
    tmin=-0.5,
    kind="average",
    nave=dual_data.shape[0],
    comment="averaged TFRs",
)

# %%
# plot topos
times = np.arange(-0, 3, 0.5)
_ = single_evoked.plot_topomap(times=times, average=0.25)

_ = dual_evoked.plot_topomap(times=times, average=0.25)

# %%
# plot average from occipital channels
occipital = ["O1", "Oz", "O2", "PO3", "POz", "PO4"]
occipial_idxs = np.array([tfr.ch_names.index(i) for i in occipital])

single_o = np.mean(single_evoked.data[occipial_idxs, ...], 0)
dual_o = np.mean(dual_evoked.data[occipial_idxs, ...], 0)

with sns.plotting_context("talk"):
    fig, ax = plt.subplots()
    ax.plot(tfr.times, single_o, label="single")
    ax.plot(tfr.times, dual_o, label="dual")
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.legend()

# %%
