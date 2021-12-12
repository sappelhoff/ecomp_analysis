"""Inspect ICA results and apply them.

Here we load the previously computed ICA solution and use several
visualizations interactively to determine which ICA components
encode blinks, horizontal eye movements, and cardiac artifacts.
For this, we will use the VEOG, HEOG, and ECG channels in the data
to inform these decisions.
We will then reject these components and apply the resulting
ICA solution to the raw, unfiltered, non-downsampled data.

Finally, this data will go through these steps:

- highpass and lowpass filtering
- interpolation of bad channels
- re-referencing to average

How to use the script?
----------------------
Either run in an interactive IPython session and have code cells rendered ("# %%")
by an editor such as VSCode, **or** run this from the command line, optionally
specifying settings as command line arguments:

```shell

python 04_inspect_ica.py --sub=1

```

"""
# %%
# Imports
import json
import multiprocessing
import sys

import matplotlib.pyplot as plt
import mne
import psutil

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_EXTERNAL, OVERWRITE_MSG
from utils import parse_overwrite

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

# overwrite existing annotation data?
overwrite = False

# interactive mode: if False, expects the excluded ICA components to be loadable
interactive = True

# filter settings
low_cutoff = 0.1
high_cutoff = 40.0

# set number of jobs for parallel processing (only if enough cores + RAM)
available_ram_gb = psutil.virtual_memory().total / 1e9
n_jobs = 1
if available_ram_gb > 16:
    n_jobs = max(n_jobs, int(multiprocessing.cpu_count() / 2))

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=sub,
        data_dir=data_dir,
        analysis_dir=analysis_dir,
        overwrite=overwrite,
        interactive=interactive,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    data_dir = defaults["data_dir"]
    analysis_dir = defaults["analysis_dir"]
    overwrite = defaults["overwrite"]
    interactive = defaults["interactive"]
    low_cutoff = defaults["low_cutoff"]
    high_cutoff = defaults["high_cutoff"]

if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

# %%
# Prepare file paths
derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
fname_fif = derivatives / f"sub-{sub:02}_concat_raw.fif.gz"

fname_fif_clean = derivatives / f"sub-{sub:02}_clean_raw.fif.gz"

fname_ica = derivatives / f"sub-{sub:02}_concat_ica.fif.gz"

savedir = analysis_dir / "derived_data" / "annotations"
fname_exclude = savedir / f"sub-{sub:02}_exclude_ica.json"

# %%
# Check overwrite
if (not overwrite) and interactive:
    for fname in [fname_fif_clean, fname_exclude]:
        if fname.exists():
            raise RuntimeError(OVERWRITE_MSG.format(fname))

# %%
# Read data
raw = mne.io.read_raw_fif(fname_fif, preload=True)
ica = mne.preprocessing.read_ica(fname_ica)

# create a copy to which to apply the ICA solution later
# (unfiltered, non-downsampled)
raw_copy = raw.copy()

# %%
# if we run in non-interactive mode, we can exit here
if not interactive:
    if not fname_exclude.exists():
        raise RuntimeError(
            f"Did not find which components to exclude for sub-{sub:02}."
        )
    with open(fname_exclude, "r") as fin:
        exclude_dict = json.load(fin)

    ica.exclude = exclude_dict["exclude"]
    raw_clean = ica.apply(inst=raw_copy.copy())
    raw_clean = raw_clean.filter(l_freq=low_cutoff, h_freq=None, n_jobs=n_jobs)
    raw_clean = raw_clean.filter(l_freq=None, h_freq=high_cutoff, n_jobs=n_jobs)
    raw_clean = raw_clean.interpolate_bads()
    raw_clean = raw_clean.set_eeg_reference(
        ref_channels="average", projection=False, ch_type="eeg"
    )
    raw_clean.save(fname_fif_clean, overwrite=overwrite)

    sys.exit()

# ... else, we continue with interactive screening of the ICA
# %%
# Preprocess the "ica raw data"
# highpass filter
raw = raw.filter(l_freq=1, h_freq=None)

# downsample
raw = raw.resample(sfreq=100)

# %%
# Automatically find artifact components using the EOG and ECG data
veog_ch_name = "Fp1" if sub == 4 else "VEOG"  # sub-04 has broken VEOG
veog_idx, veog_scores = ica.find_bads_eog(raw, veog_ch_name)
heog_idx, heog_scores = ica.find_bads_eog(raw, "HEOG")
ecg_idx, ecg_scores = ica.find_bads_ecg(raw, "ECG")

# %%
# Plot ICA components
with plt.style.context("default"):
    fig = ica.plot_components(inst=raw)

# %%
# Create VEOG epochs and plot evoked
epochs_veog = mne.preprocessing.create_eog_epochs(
    raw, ch_name=veog_ch_name, picks="eeg"
)
fig = epochs_veog.average().plot()

# %%
# Create HEOG epochs and plot evoked
epochs_heog = mne.preprocessing.create_eog_epochs(raw, ch_name="HEOG", picks="eeg")
fig = epochs_heog.average().plot()

# %%
# Create ECG epochs and plot evoked
epochs_ecg = mne.preprocessing.create_ecg_epochs(raw, ch_name="ECG", picks="eeg")
fig = epochs_ecg.average().plot()

# %%
# Plot VEOG scores and overlay
exclude_veog = veog_idx
fig = ica.plot_scores(
    veog_scores, exclude=exclude_veog, title=f"VEOG, exclude: {exclude_veog}"
)
fig = ica.plot_overlay(epochs_veog.average(), exclude=exclude_veog, show=False)
fig.tight_layout()

# %%
# Plot HEOG scores and overlay
exclude_heog = heog_idx
fig = ica.plot_scores(
    heog_scores, exclude=exclude_heog, title=f"HEOG, exclude: {exclude_heog}"
)
fig = ica.plot_overlay(epochs_heog.average(), exclude=exclude_heog, show=False)
fig.tight_layout()

# %%
# Plot ECG scores and overlay
exclude_ecg = ecg_idx
fig = ica.plot_scores(
    ecg_scores, exclude=exclude_ecg, title=f"ECG, exclude: {exclude_ecg}"
)
fig = ica.plot_overlay(epochs_ecg.average(), exclude=exclude_ecg, show=False)
fig.tight_layout()

# %%
# Set ica.exclude attribute
ica.exclude = list(set(exclude_veog + exclude_heog + exclude_ecg))
exclude = [int(i) for i in ica.exclude]  # convert numpy.int64 -> int
assert exclude == ica.exclude
print(f"Excluding: {exclude}")

with open(fname_exclude, "w") as fout:
    json.dump(dict(exclude=exclude), fout, indent=4)
    fout.write("\n")

# %%
# Apply ICA to raw data
# Exclude components from ica.exclude
raw_clean = ica.apply(inst=raw_copy.copy())

# %%
# Preprocess ica-cleaned data and save
# Highpass filtering
raw_clean = raw_clean.filter(l_freq=low_cutoff, h_freq=None, n_jobs=n_jobs)

# Lowpass filtering
raw_clean = raw_clean.filter(l_freq=None, h_freq=high_cutoff, n_jobs=n_jobs)

# Interpolation
raw_clean = raw_clean.interpolate_bads()

# Re-referencing
raw_clean = raw_clean.set_eeg_reference(
    ref_channels="average", projection=False, ch_type="eeg"
)

# %%
# Visually screen clean data for a final sanity check
with mne.viz.use_browser_backend("pyqtgraph"):
    raw_clean.plot(
        block=True,
        use_opengl=False,
        n_channels=len(raw_clean.ch_names),
        bad_color="red",
        duration=20.0,
        clipping=None,
    )

# %%
# Save as cleaned data
raw_clean.save(fname_fif_clean, overwrite=overwrite)

# %%
