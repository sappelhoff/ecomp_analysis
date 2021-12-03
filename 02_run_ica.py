"""Run ICA on the raw data.

In this script we run extended infomax ICA on each participant data and save the
resulting MNE-Python ICA object for later inspection.

This script takes some time to run, depending on how fast your CPU is,
and how many cores and RAM are/is available. (Takes about 10 minutes per
participant on an Intel® Core™ i7-8650U CPU @ 1.90GHz × 8 with 32GB RAM)
We recommend to either:

- run this over night on a fast machine (or server)
- use the already saved ICA data (if available and overwrite is set to False,
  this script will do nothing and you should run the next script)
- adjust the script to run ICA only on a subspace (using PCA) or temporal subset
  (using more aggressive downsampling or by only supplying portions) of the data

Note that in the last case, you MUST inspect the resulting ICA components again. Using
the saved component indices (if available) to reject components will lead to wrong
results unless they are used with the saved ICA data. Hints on how to speed up ICA
computations are given below in the code comments.

We do the following steps:

- load concatenated raw data
- preprocessing the data for ICA
    - highpass filtering the data at 1Hz
    - downsampling the data to 100Hz (applying appropriate lowpass filter to
      prevent aliasing)
- running ICA on non-broken EEG channels only (using an RNG seed)
- Save the ICA object

"""
# %%
# Imports
import pathlib
import sys

import click
import mne

from config import BAD_SUBJS, DATA_DIR_EXTERNAL

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source here
data_dir = DATA_DIR_EXTERNAL

# overwrite existing annotation data?
overwrite = False

# random number generator seed for the ICA
ica_rng = 42

# %%


# Potentially overwrite settings with command line arguments
@click.command()
@click.option("-s", "--sub", default=sub, type=int, help="Subject number")
@click.option("-d", "--data_dir", default=data_dir, type=str, help="Data location")
@click.option("-o", "--overwrite", default=overwrite, type=bool, help="Overwrite?")
@click.option("-r", "--ica_rng", default=ica_rng, type=int, help="ICA seed")
def get_inputs(sub, data_dir, overwrite, ica_rng):
    """Parse inputs in case script is run from command line."""
    print("Overwriting settings from command line.\nUsing the following settings:")
    for name, opt in [
        ("sub", sub),
        ("data_dir", data_dir),
        ("overwrite", overwrite),
        ("ica_rng", ica_rng),
    ]:
        print(f"    > {name}: {opt}")

    data_dir = pathlib.Path(data_dir)
    return sub, data_dir, overwrite, ica_rng


# only run this when not in an IPython session
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    sub, data_dir, overwrite, ica_rng = get_inputs.main(standalone_mode=False)

# %%
# Prepare file paths
derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
fname_fif = derivatives / f"sub-{sub:02}_concat_raw.fif.gz"

fname_ica = derivatives / f"sub-{sub:02}_concat_ica.fif.gz"

# %%
# Check overwrite
overwrite_msg = "\nfile exists and overwrite is False:\n\n>>> {}\n"
if fname_ica.exists() and not overwrite:
    raise RuntimeError(overwrite_msg.format(fname_ica))

if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

# %%
# Load raw data
raw = mne.io.read_raw_fif(fname_fif, preload=True)

# %%
# Preprocess raw data copy for ICA
# highpass filter
raw = raw.filter(l_freq=1, h_freq=None)

# downsample
raw = raw.resample(sfreq=100)

# %%
# Initialize an ICA object, using extended infomax
# To speed up the computations, consider passing:
# n_compontents=32
# method="picard"
# fit_params=dict(ortho=False, extended=True)
assert len(raw.info["projs"]) == 0
ica = mne.preprocessing.ICA(
    n_components=None,
    random_state=ica_rng,
    method="infomax",
    fit_params=dict(extended=True),
)
# %%
# Get the channel indices of all channels that are *clean* and of type *eeg*
all_idxs = list(range(len(raw.ch_names)))

bad_idxs = [raw.ch_names.index(ii) for ii in raw.info["bads"]]
eog_idxs = [raw.ch_names.index(ii) for ii in raw.ch_names if "EOG" in ii]
ecg_idxs = [raw.ch_names.index(ii) for ii in raw.ch_names if "ECG" in ii]

exclude_idxs = bad_idxs + eog_idxs + ecg_idxs
ica_idxs = list(set(all_idxs) - set(exclude_idxs))

# Fit our raw (high passed, downsampled) data with our ica object
# To speed up the computations, consider passing:
# start=np.percentile(raw.times, 25)
# stop=np.percentile(raw.times, 75)
ica.fit(
    inst=raw,
    picks=ica_idxs,
    reject_by_annotation=True,
    start=None,
    stop=None,
)
# %%
# Save the ica object
ica.save(fname_ica, overwrite=overwrite)
