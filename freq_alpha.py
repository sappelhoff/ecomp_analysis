"""Analyze alpha power in the single vs dual tasks.

First read raw data.
Then make epochs of whole trials:

1. fixstim shown for 500ms (trigger code single: 1, dual: 101)
2. fixstim offset (trigger code single: 2, dual: 102)
3. 10 samples shown for 350ms each (different trigger codes)
--> make epochs centered on fixstim offset with time -500ms - 3500ms

Then compute alpha power per over trials, per task (single vs dual)
Save the resulting TFR object as HDF5 file.

"""
# %%
# Imports
import sys

import mne
import numpy as np
from mne_faster import find_bad_epochs

from config import BAD_SUBJS, DATA_DIR_EXTERNAL, FASTER_THRESH, STREAMS
from utils import parse_overwrite

# %%
# Settings
data_dir = DATA_DIR_EXTERNAL
sub = 1
overwrite = False

downsample_freq = 250
t_min_max_epochs = (-0.5, 3.5)

baseline = (None, 0)  # irrelevant prior to TFR

# TODO
# cycles: use fixed number for all freqs ... e.g., 7
# use decim to downsample TFR ... 50ms steps?
# morlet: is it forwards and backwards? ... check in mne docs
# baselining: AFTER tfr --> but not needed when single minus dual
# check peak: what happens in epoch at 0.5s?
# Make epochs longer, then crop?
#

# alpha power ~8-14 Hz
freqs = np.arange(8, 15)

# make time/freq resolution frequency band dependent
# see also:
# https://www.fieldtriptoolbox.org/tutorial/timefrequencyanalysis/#morlet-wavelets
n_cycles = freqs / 2.0

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=sub,
        data_dir=data_dir,
        overwrite=overwrite,
    )

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    data_dir = defaults["data_dir"]
    overwrite = defaults["overwrite"]

if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"
fname_fif_clean_template = str(
    derivatives / "sub-{sub:02}" / "sub-{sub:02}_clean_raw.fif.gz"
)

fname_tfr_template = str(
    derivatives / "sub-{sub:02}" / "sub-{sub:02}_stream-{stream}_tfr.h5"
)

# %%
# Read raw
fname = fname_fif_clean_template.format(sub=sub)
raw = mne.io.read_raw_fif(fname, preload=True)

# %%
# Get events: pick only "fixstim offset" in trial
# see ecomp_experiment/define_ttl.py
# 2 -> fixstim offset in "single stream"
# 102 -> fixstim offset in "dual stream"
event_id = {"Stimulus/S  2": 2, "Stimulus/S102": 102}

events, event_id = mne.events_from_annotations(raw, event_id=event_id)
assert events.shape[0] == 600  # 2 stream, 300 trials, 1 fixstim offset per trial

# Make event_id human readable
event_id_human = {"single": 2, "dual": 102}

# %%
# Epoch the data

# Decimate the data instead of resampling (i.e., taking every nth sample).
# Data is already filtered at this point, so no aliasing artifacts occur
assert raw.info["lowpass"] <= (downsample_freq / 3)
assert raw.info["sfreq"] == 1000  # as recorded
assert raw.info["sfreq"] % downsample_freq == 0
decim = int(raw.info["sfreq"] / downsample_freq)

# unpack tmin tmax from tuple
tmin, tmax = t_min_max_epochs

# make epochs
epochs = mne.Epochs(
    raw=raw,
    events=events,
    event_id=event_id_human,
    decim=decim,  # based on downsample_freq
    preload=True,
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    picks=["eeg"],  # we won't need the EOG and ECG channels from here on
    reject_by_annotation=True,
)

# %%
# Distinguish single and dual stream data
epo_dict = {}
for stream in STREAMS:
    epo_dict[stream] = epochs[stream]


# %%
# drop epochs automatically according to FASTER pipeline, step 2
for stream, epo in epo_dict.items():
    bad_epos = find_bad_epochs(epo, thres=FASTER_THRESH)
    epo.drop(bad_epos, reason="FASTER_AUTOMATIC")

# %%
# Do TFR decomposition separately for each stream and save
tfr_dict = {}
for stream, epo in epo_dict.items():

    tfr_dict[stream] = mne.time_frequency.tfr_morlet(
        epo,
        freqs,
        n_cycles,
        average=True,
        return_itc=False,
        use_fft=False,
        picks=["eeg"],
        n_jobs=1,
        verbose=True,
    )

    fname = fname_tfr_template.format(sub=sub, stream=stream)
    tfr_dict[stream].save(fname, overwrite)

# %%
