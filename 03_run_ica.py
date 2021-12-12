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

How to use the script?
----------------------
Either run in an interactive IPython session and have code cells rendered ("# %%")
by an editor such as VSCode, **or** run this from the command line, optionally
specifying settings as command line arguments:

```shell

python 03_run_ica.py --sub=1

```

"""
# %%
# Imports
import multiprocessing
import sys

import mne
import psutil

from config import BAD_SUBJS, DATA_DIR_EXTERNAL, DEFAULT_RNG_SEED, OVERWRITE_MSG
from utils import parse_overwrite

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source here
data_dir = DATA_DIR_EXTERNAL

# overwrite existing annotation data?
overwrite = False

# random number generator seed for the ICA
ica_rng = DEFAULT_RNG_SEED

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
        overwrite=overwrite,
        ica_rng=ica_rng,
    )

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    data_dir = defaults["data_dir"]
    overwrite = defaults["overwrite"]
    ica_rng = defaults["ica_rng"]

if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

# %%
# Prepare file paths
derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
fname_fif = derivatives / f"sub-{sub:02}_concat_raw.fif.gz"

fname_ica = derivatives / f"sub-{sub:02}_concat_ica.fif.gz"

# %%
# Check overwrite
if fname_ica.exists() and not overwrite:
    raise RuntimeError(OVERWRITE_MSG.format(fname_ica))

# %%
# Load raw data
raw = mne.io.read_raw_fif(fname_fif, preload=True)

# %%
# Preprocess raw data copy for ICA
# highpass filter
raw = raw.filter(l_freq=1, h_freq=None, n_jobs=n_jobs)

# downsample
raw = raw.resample(sfreq=100, n_jobs=n_jobs)

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
