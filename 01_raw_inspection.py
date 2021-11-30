"""Inspect the raw data.

In this script we go through each participant and prepare the data for preprocessing
with ICA. We will mainly prepare the BrainVision data to be in an MNE-Python accessible
format and screen it for bad channels and segments.

- For each subject:
    - Load data from both tasks and concatenate to a single raw file
    - anonymize the file (see below)
    - Set channel types (EEG, EOG, ECG) and add electrode positions (standard montage)
    - Get all annotations, and remove the BrainVision ones ("New Segment", ...)
    - Save these annotations in a variable ("annots_orig")
    - Remove existing annotations from raw, for browsing the data without them
    - Automatically create some new annotations (block breaks, muscle scores; see below)
    - screen data for bad channels and segments, double checking the autocreated ones
    - Save bad channels and segments
    - Get union of "annots_orig" and newly created annotations and add to raw
    - Save raw with full annotations (original + bad) and bad channels in FIF format

Anonymization
-------------
To anonymize the file, we change the recording date by subtracting a certain number of
days, read from a local file called `DAYSBACK.json`. If that file is not available
on your system, the recording dates will remain unchanged. To make sure that the
anonymization serves its purpose, `DAYSBACK.json` is not shared publicly.

Automatic marking of bad segments
---------------------------------
We use three methods to "pre-mark" segments as bad, and then verify the results in
an interactive screening:

- we know when a participant was having breaks, we mark these as bad_break
- mne.preprocessing.annotate_flat helps us to mark flat segments (and channels)
- mne.preprocessing.annotate_muscle_zscore helps us mark segments with muscle activity

We let these functions mostly operate with their default values, and double check the
results.


"""

# %%
# Imports
import mne
import numpy as np

from config import ANALYSIS_DIR, DATA_DIR_EXTERNAL, get_daysback, get_sourcedata

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source here
data_dir = DATA_DIR_EXTERNAL

# overwrite existing annotation data?
overwrite = False

# %%
# Load and concatenate data
raws = []
for stream in ["single", "dual"]:
    vhdr, tsv = get_sourcedata(1, stream, data_dir)
    raw_stream = mne.io.read_raw_brainvision(vhdr, preload=True)
    raws.append(raw_stream)

raw = mne.concatenate_raws(raws)

# %%
# Anonymize data
daysback = get_daysback(data_dir)
raw = raw.anonymize(daysback=daysback, keep_his=False, verbose=False)

# %%
# Prepare raw object (ch_types, montage, ...)
# Set the EOG and ECG channels to their type
raw = raw.set_channel_types({"ECG": "ecg", "HEOG": "eog", "VEOG": "eog"})

# Set a standard montage for plotting later
montage = mne.channels.make_standard_montage("standard_1020")
raw = raw.set_montage(montage)

# Add some recording info
raw.info["line_freq"] = 50

# %%
# Handle original annotations
annots_orig = raw.annotations

bv_annots_to_remove = ["Comment/ControlBox is not connected via USB", "New Segment/"]
idxs_to_remove = []
for annot in bv_annots_to_remove:
    idxs = np.nonzero(annots_orig.description == annot)[0]
    idxs_to_remove.append(idxs)

annots_orig.delete(idxs_to_remove)

# remove annotations from raw, we'll add some more on a blank slate now
raw = raw.set_annotations(None, verbose=False)

# %%
# Automatically add annotations on bad segments
# mark flat segments
annots_flat, bads = mne.preprocessing.annotate_flat(raw, picks="eeg", min_duration=0.5)
raw.info["bads"] += bads


# Bad "break" segments
def get_stim_onset(annots, stim, nth_stim=0):
    """Help to find onset of a stimulus in the data."""
    idx = (annots.description == stim).nonzero()[0][nth_stim]
    return idx, annots.onset[idx]


annots_break = mne.Annotations([], [], [])
for stream in ["single", "dual"]:
    # marker meanings, see ecomp_experiment package
    ttl_start = "Stimulus/S 80" if stream == "single" else "Stimulus/S180"
    ttl_break_begin = "Stimulus/S  7" if stream == "single" else "Stimulus/S107"
    ttl_break_end = "Stimulus/S  8" if stream == "single" else "Stimulus/S108"

    # Mark data prior to start bad
    _, recording_onset = get_stim_onset(annots_orig, ttl_start)
    tbuffer = 1
    start = raw.first_samp / raw.info["sfreq"]
    dur = (recording_onset - start) - tbuffer
    annots_break.append(start, dur, "BAD_break")

    # Mark block breaks bad
    nbreaks = 6
    tbuffer = 1
    for ith_break in range(nbreaks):
        _, onset = get_stim_onset(annots_orig, ttl_break_begin, nth_stim=ith_break)
        _, offset = get_stim_onset(annots_orig, ttl_break_end, nth_stim=ith_break)

        begin = onset + tbuffer
        dur = (offset - tbuffer) - begin
        annots_break.append(begin, dur, "BAD_break")

    # Mark data from last block break offset to end bad
    dur = (raw.last_samp / raw.info["sfreq"]) - offset
    annots_break.append(offset, dur, "BAD_break")


# bad muscle segments
raw_copy = raw.copy()
raw_copy.notch_filter([50, 100])

threshold_muscle = 6
annots_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
    raw_copy, threshold=threshold_muscle, ch_type="eeg"
)

# combine all automatically identified bad segments
# TODO: COMBINE ANNOTS
annots_bad = 1
raw.set_annotations(annots_bad)
# %%
# Screen the data

# interactive plot
with mne.viz.use_browser_backend("pyqtgraph"):
    raw.plot(
        block=True,
        use_opengl=False,
        n_channels=len(raw.ch_names),
        bad_color="red",
        duration=20.0,
        clipping=None,
    )

# %%
# Save results

savedir = ANALYSIS_DIR / "derived_data" / "annotations"
savedir.mkdir(parents=True, exist_ok=True)

overwrite_msg = "\nfile exists and overwrite is False:\n\n>>> {}\n"

# Save bad channels
fname_channels = savedir / f"sub-{sub:02}_bad-channels.txt"
if not fname_channels.exists() or overwrite:
    with open(fname_channels, "w") as fout:
        lines = "\n".join(raw.info["bads"])
        fout.writelines(lines)
else:
    raise RuntimeError(overwrite_msg.format(fname_channels))

# Save bad annotations
fname_annots = savedir / f"sub-{sub:02}_annotations.txt"
if not fname_annots.exists() or overwrite:
    raw.annotations.save(fname_annots)
else:
    raise RuntimeError(overwrite_msg.format(fname_annots))

# Save data with all annots as FIF
derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
derivatives.mkdir(parents=True, exist_ok=True)
fname_fif = derivatives / f"sub-{sub:02}_concat_raw.fif.gz"

all_annots = annots_orig + raw.annotations
raw.set_annotations(all_annots)

if not fname_fif.exists() or overwrite:
    raw.save(fname_fif)
else:
    raise RuntimeError(overwrite_msg.format(fname_fif))
