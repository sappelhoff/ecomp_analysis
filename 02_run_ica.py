"""Run ica on the raw data.

In this script we run extended infomax ICA on each participant data and save the
resulting MNE-Python ICA object for later inspection.

This script takes some time to run, depending on how fast your CPU is,
and how many cores and RAM are/is available.
We recommend to either:

- run this over night on a fast machine (or server)
- use the already saved ICA data (if available); done automatically if `overwrite=False`
  below
- adjust the script to run ICA only on a subspace (using PCA) or subset
  (using more aggressive downsampling or by only supplying portions) of the data

Note that in the last case, you MUST inspect the resulting ICA components again. Using
the saved component indices to reject (if available) will lead to wrong results unless
they are used with the saved ICA data.



- load concatenated raw data
- preprocessing the data for ICA
    - highpass filtering the data at 1Hz
    - downsampling the data to 250Hz (applying appropriate lowpass filter to prevent aliasing)
- running ICA on non-broken EEG channels only
- Save the ICA object


"""
# %%
# Imports
import pathlib
import sys

import click
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

from config import ANALYSIS_DIR, BAD_SUBJS, DATA_DIR_EXTERNAL

# %%
# Settings
# Select the subject to work on here
sub = 4

# Select the data source here
data_dir = DATA_DIR_EXTERNAL

# overwrite existing annotation data?
overwrite = False

# interactive mode: if False, expects the annotations to be loadable
interactive = True

# random number generator seed for the ICA
ica_rng = 42

# %%


# Potentially overwrite settings with command line arguments
@click.command()
@click.option("-s", "--sub", default=sub, type=int, help="Subject number")
@click.option("-d", "--data_dir", default=data_dir, type=str, help="Data location")
@click.option("-o", "--overwrite", default=overwrite, type=bool, help="Overwrite?")
@click.option("-i", "--interactive", default=interactive, type=bool, help="Interative?")
@click.option("--ica_rng", default=ica_rng, type=bool, help="ICA seed")
def get_inputs(sub, data_dir, overwrite, interactive, ica_rng):
    """Parse inputs in case script is run from command line."""
    print("Overwriting settings from command line.\nUsing the following settings:")
    for name, opt in [
        ("sub", sub),
        ("data_dir", data_dir),
        ("overwrite", overwrite),
        ("interactive", interactive),
        ("ica_rng", ica_rng),
    ]:
        print(f"    > {name}: {opt}")

    data_dir = pathlib.Path(data_dir)
    return sub, data_dir, overwrite, interactive


# only run this when not in an IPython session
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    sub, data_dir, overwrite, interactive = get_inputs.main(standalone_mode=False)


# %%
# Prepare file paths
derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
fname_fif = derivatives / f"sub-{sub:02}_concat_raw.fif.gz"

# %%
# Load raw data and work on a copy
raw_concat = mne.io.read_raw_fif(fname_fif, preload=True)
raw = raw_concat.copy()

# %%
# Preprocess raw data copy for ICA
raw = raw.filter(l_freq=1, h_freq=30)

# downsample
raw = raw.resample(sfreq=100)

# %%
# Run ICA
# Initialize an ICA object, using extended infomax
assert len(raw.info['projs']) == 0
ica = mne.preprocessing.ICA(
    random_state=ica_rng, method="infomax", fit_params=dict(extended=True)
)
# %%
ica = mne.preprocessing.ICA(random_state=ica_rng,
                            method="picard", fit_params=dict(ortho=False, extended=True)
                            )
# %%
# Get the channel indices of all channels that are *clean* and of type *eeg*
all_idxs = list(range(len(raw.ch_names)))

bad_idxs = [raw.ch_names.index(ii) for ii in raw.info["bads"]]
eog_idxs = [raw.ch_names.index(ii) for ii in raw.ch_names if "EOG" in ii]
ecg_idxs = [raw.ch_names.index(ii) for ii in raw.ch_names if "ECG" in ii]

exclude_idxs = bad_idxs + eog_idxs + ecg_idxs
ica_idxs = list(set(all_idxs) - set(exclude_idxs))

# Fit our raw (high passed, downsampled) data to our ica object
# fit only on short middle part of data, to avoid long fitting times
ica.fit(raw, picks=ica_idxs, reject_by_annotation=True,
        start=np.percentile(raw.times, 25), stop=np.percentile(raw.times, 75))
#%%

# Automatically find artifact components using the EOG and ECG data
veog_idx, veog_scores = ica.find_bads_eog(raw, "VEOG")
heog_idx, heog_scores = ica.find_bads_eog(raw, "HEOG")
ecg_idx, ecg_scores = ica.find_bads_ecg(raw, "ECG")

# exclude the automatically identified components
ica.exclude = list(set(veog_idx + heog_idx + ecg_idx))

# %%
ica.exclude
# %%


matplotlib.pyplot.switch_backend("Qt5Agg")

ica.plot_components(inst=raw)
# %%

# %%
# %%
epochs_veog = mne.preprocessing.create_eog_epochs(raw, ch_name="VEOG")
events_veog = epochs_veog.events
epochs_veog.average().plot()
# %%
epochs_heog = mne.preprocessing.create_eog_epochs(raw, ch_name="HEOG")
events_heog = epochs_heog.events
epochs_heog.average().plot()

# %%
epochs_ecg = mne.preprocessing.create_ecg_epochs(raw, ch_name="ECG")
events_ecg = epochs_ecg.events
epochs_ecg.average().plot()

# %%
ica.plot_overlay(epochs_veog.average(), exclude=veog_idx, show=False)
# %%
ica.plot_overlay(epochs_heog.average(), exclude=heog_idx, show=False)
# %%
ica.plot_overlay(epochs_ecg.average(), exclude=[12], show=False)
# %%

ica.plot_scores(veog_scores, exclude=veog_idx, title="VEOG")
#%%
ica.plot_scores(heog_scores, exclude=heog_idx, title="HEOG")
#%%
ica.plot_scores(ecg_scores, exclude=[12], title="ECG")
# %%
# %%
fig, ax = plt.subplots()
raw_concat.plot_psd(picks="eeg", fmax=110, average=True, ax=ax)
ax.axvline(10, color="gray", linestyle="-.")

# %%
report = mne.Report(title='Figure example')
report.add_figure(
    fig=fig, title='A custom figure',
    caption='A blue dashed line reaches up into the sky …',
    image_format='PNG'
)

events, event_id = mne.events_from_annotations(raw_concat)
report.add_events(events, event_id=event_id, title="NONO", sfreq=raw_concat.info["sfreq"])

report.add_raw(raw_concat, title="ye", psd=True,)

fname = '/home/stefanappelhoff/Desktop/report_custom_figure.html'
report.save(fname, open_browser=False, overwrite=True)


# %%
