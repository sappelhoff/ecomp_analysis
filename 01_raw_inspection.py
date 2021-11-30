"""Inspect the raw data."""

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np

from config import DATA_DIR_EXTERNAL, get_sourcedata

# %%
# Load and prepare data
stream = "single"
vhdr, tsv = get_sourcedata(1, stream, DATA_DIR_EXTERNAL)
raw = mne.io.read_raw_brainvision(vhdr, preload=True)

# Set the EOG and ECG channels to their type
raw.set_channel_types({"ECG": "ecg", "HEOG": "eog", "VEOG": "eog"})

# Set a standard montage for plotting later
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# Temporarily remove existing annotations for faster interactive plotting
# raw.set_annotations(None)

# %%
# Temporarily remove existing annotations for faster interactive plotting
raw.set_annotations(None)

with mne.viz.use_browser_backend("pyqtgraph"):
    raw.plot(block=True, use_opengl=False)

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

# %%

# Mark muscles
raw_copy = raw.copy()
raw_copy.notch_filter([50, 100])

threshold_muscle = 6
annots_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
    raw_copy, threshold=threshold_muscle, ch_type="eeg"
)

# %%
fig, ax = plt.subplots()
ax.plot(raw.times, scores_muscle)
ax.axhline(y=threshold_muscle, color="r")
ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")

# %%
raw_copy.set_annotations(annots_muscle)
raw_copy.plot(block=True)
# %%

# mark flat

annots_flat, bads = mne.preprocessing.annotate_flat(raw, picks="eeg", min_duration=1)

# %%

# TODO: downsample before plotting
fig = raw.plot(
    n_channels=len(raw.ch_names), bad_color=(1, 0, 0), duration=20.0, block=True
)

# save
# union of annotations
# bad channels
# threshold for auto muscle detection
