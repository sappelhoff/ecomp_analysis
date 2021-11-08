"""Inspect the raw data."""

# %%
import mne
import numpy as np

from config import get_sourcedata

# %%
# Load and prepare data
stream = "single"
vhdr, tsv = get_sourcedata(1, stream)
raw = mne.io.read_raw_brainvision(vhdr, preload=True)

# Set the EOG and ECG channels to their type
raw.set_channel_types({"ECG": "ecg", "HEOG": "eog", "VEOG": "eog"})

# Set a standard montage for plotting later
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# Temporarily remove existing annotations for faster interactive plotting
# raw.set_annotations(None)

# %%


def get_stim_onset(raw, stim, nth_stim=0):
    """Help to find onset of a stimulus in the data."""
    idx = (raw.annotations.description == stim).nonzero()[0][nth_stim]
    return idx, raw.annotations.onset[idx]


# %%


np.unique(raw.annotations.description)

# Remove BrainVision Recorder automated annotations
idxs = np.nonzero(
    [
        i in ["Comment/ControlBox is not connected via USB", "New Segment/"]
        for i in raw.annotations.description
    ]
)[0]
raw.annotations.delete(idxs)

# %%


annots = mne.Annotations([], [], [])

# marker meanings, see ecomp_experiment package
ttl_start = "Stimulus/S 80" if stream == "single" else "Stimulus/S180"
ttl_break_begin = "Stimulus/S  7" if stream == "single" else "Stimulus/S107"
ttl_break_end = "Stimulus/S  8" if stream == "single" else "Stimulus/S108"

# Mark data prior to start bad
_, recording_onset = get_stim_onset(raw, ttl_start)
tbuffer = 1
start = raw.first_samp / raw.info["sfreq"]
dur = (recording_onset - start) - tbuffer
annots.append(start, dur, "BAD_break")

# Mark block breaks bad
nbreaks = 6
tbuffer = 1
for ith_break in range(nbreaks):
    _, onset = get_stim_onset(raw, ttl_break_begin, nth_stim=ith_break)
    _, offset = get_stim_onset(raw, ttl_break_end, nth_stim=ith_break)

    begin = onset + tbuffer
    dur = (offset - tbuffer) - begin
    annots.append(begin, dur, "BAD_break")

# Mark data from last block break offset to end bad
dur = (raw.last_samp / raw.info["sfreq"]) - offset
annots.append(offset, dur, "BAD_break")
