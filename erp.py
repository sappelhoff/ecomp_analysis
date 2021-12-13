"""Compute and plot ERPs over subjects."""
# %%
# Imports

import mne
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_EXTERNAL

# %%

data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

# %%
subjects = np.array(list(set(range(1, 33)) - set(BAD_SUBJS)))
subjects
baseline = (None, 0)
# %%
derivatives = data_dir / "derivatives"

epochs_list = []
for sub in tqdm(subjects):

    fname_epo = derivatives / f"sub-{sub:02}" / f"sub-{sub:02}_numbers_epo.fif.gz"

    epochs = mne.read_epochs(fname_epo, preload=True, verbose=False)
    epochs = epochs.apply_baseline(baseline, verbose=False)
    epochs_list += [epochs]


# %%
numbers = range(1, 10)
evoked_dict = {str(i): [] for i in numbers}
mean_amps = []
for number in tqdm(evoked_dict):

    for epo in epochs_list:
        evoked = epo[f"{number}"].average()

        # evoked.data = (evoked.data.T - evoked.data.mean(axis=1)).T
        # evoked.data = evoked.data - evoked.data.mean(axis=0)
        evoked.data = evoked.data - epo.average().data

        evoked_dict[number] += [evoked]


# %%
cmap = sns.color_palette("crest_r", as_cmap=True)
p3_group = ["Cz", "C1", "C2", "CPz", "CP1", "CP2", "CP3", "CP4", "Pz", "P1", "P2"]
mne.viz.plot_compare_evokeds(
    evoked_dict, picks=p3_group, combine="mean", show_sensors=True, cmap=cmap, ci=0.68
)
# %%


# %%
