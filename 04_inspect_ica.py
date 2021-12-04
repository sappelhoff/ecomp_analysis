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

"""
# %%
# Imports
import pathlib
import sys

import click
import mne

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_EXTERNAL

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

# overwrite existing annotation data?
overwrite = False

# filter settings
low_cutoff = 0.1
high_cutoff = 40.0

# %%


# Potentially overwrite settings with command line arguments
@click.command()
@click.option("-s", "--sub", default=sub, type=int, help="Subject number")
@click.option("-d", "--data_dir", default=data_dir, type=str, help="Data location")
@click.option("-a", "--analysis_dir", default=data_dir, type=str, help="Analysis dir")
@click.option("-o", "--overwrite", default=overwrite, type=bool, help="Overwrite?")
@click.option("-l", "--low_cutoff", default=low_cutoff, type=float, help="low_cutoff")
@click.option(
    "-h", "--high_cutoff", default=high_cutoff, type=float, help="high_cutoff"
)
def get_inputs(sub, data_dir, analysis_dir, overwrite, low_cutoff, high_cutoff):
    """Parse inputs in case script is run from command line."""
    print("Overwriting settings from command line.\nUsing the following settings:")
    for name, opt in [
        ("sub", sub),
        ("data_dir", data_dir),
        ("analysis_dir", data_dir),
        ("overwrite", overwrite),
        ("low_cutoff", low_cutoff),
        ("high_cutoff", high_cutoff),
    ]:
        print(f"    > {name}: {opt}")

    data_dir = pathlib.Path(data_dir)
    analysis_dir = pathlib.Path(analysis_dir)
    return sub, data_dir, analysis_dir, overwrite, low_cutoff, high_cutoff


# only run this when not in an IPython session
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    sub, data_dir, analysis_dir, overwrite = get_inputs.main(standalone_mode=False)

# %%
# Prepare file paths
derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
fname_fif = derivatives / f"sub-{sub:02}_concat_raw.fif.gz"

fname_fif_clean = derivatives / f"sub-{sub:02}_clean_raw.fif.gz"

fname_ica = derivatives / f"sub-{sub:02}_concat_ica.fif.gz"

# %%
# TODO: interactive way to run
print(analysis_dir)

# %%
# Read data
if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

raw = mne.io.read_raw_fif(fname_fif, preload=True)
ica = mne.preprocessing.read_ica(fname_ica)

# create a copy to which to apply the ICA solution later
# (unfiltered, non-downsampled)
raw_copy = raw.copy()

# %%
# Preprocess the "ica raw data"
# highpass filter
raw = raw.filter(l_freq=1, h_freq=None)

# downsample
raw = raw.resample(sfreq=100)

# %%
# Automatically find artifact components using the EOG and ECG data
veog_idx, veog_scores = ica.find_bads_eog(raw, "VEOG")
heog_idx, heog_scores = ica.find_bads_eog(raw, "HEOG")
ecg_idx, ecg_scores = ica.find_bads_ecg(raw, "ECG")

# %%
# Plot ICA components
fig = ica.plot_components(inst=raw)

# %%
# Create VEOG epochs and plot evoked
epochs_veog = mne.preprocessing.create_eog_epochs(raw, ch_name="VEOG", picks="eeg")
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
print(f"Excluding: {ica.exclude}")

# %%
# Apply ICA to raw data
# Exclude components from ica.exclude
raw_clean = ica.apply(inst=raw_copy.copy())

# %%
# Preprocess ica-cleaned data and save
# Highpass filtering
raw_clean = raw_clean.filter(l_freq=low_cutoff, h_freq=None)

# Lowpass filtering
raw_clean = raw_clean.filter(l_freq=None, h_freq=high_cutoff)

# Interpolation
raw_clean = raw_clean.interpolate_bads()

# Re-referencing
raw_clean = raw_clean.set_eeg_reference(
    ref_channels="average", projection=False, ch_type="eeg"
)

# Save as cleaned data
raw_clean.save(fname_fif_clean)

# %%
