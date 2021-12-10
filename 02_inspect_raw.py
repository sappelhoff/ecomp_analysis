"""Inspect the raw data.

In this script we go through each participant and prepare the data for preprocessing
with ICA. We will mainly prepare the BrainVision data to be in an MNE-Python accessible
format and screen it for bad channels and segments.

We can also run this in non-interactive mode, by setting `interactive=False`. In that
case, the script will search for already-saved annotation files, apply them to the
raw data object, and save the data object as a FIF file. This is helpful when
only the raw data and annotation files have been shared and you want to re-create
the FIF files.

In general, this is the workflow:

- prepare the data, this includes the following steps:
    - Load data from both tasks and concatenate to a single raw file
    - anonymize the file
    - Set channel types (EEG, EOG, ECG) and add electrode positions (standard montage)
    - Set line noise frequency to 50Hz
- Get all annotations, and remove the BrainVision ones ("New Segment", ...)
- Save these annotations in a variable ("annots_orig")
- Remove existing annotations from raw, for browsing the data without them
- Automatically mark some channels as bad (see below)
- Automatically create some new annotations (block breaks, ...; see below)
- screen data for bad channels and segments, double checking the autocreated ones
- Save bad channels and segments
- Get union of "annots_orig" and newly created annotations and add to raw
- Save raw with full annotations (original + bad) and bad channels in FIF format

How to use the script?
----------------------
Either run in an interactive IPython session and have code cells rendered ("# %%")
by an editor such as VSCode, **or** run this from the command line, optionally
specifying settings as command line arguments:

```shell

python 02_inspect_raw.py --sub=1

```

Automatic marking of bad channels
---------------------------------
The outputs from the "01_find_bads.py" script are optionally read in and "pre-marked"
in the raw data as bad channels. In the interactive inspection of the data these
automatically marked bad channels should then be double checked and either kept
as bad, or revised to be "good" instead.

Automatic marking of bad channels happens via the "pyprep" Python package.

Automatic marking of bad segments
---------------------------------
We use three methods to "pre-mark" segments as bad, and then verify the results in
an interactive screening:

- mne.preprocessing.annotate_break to mark block breaks
- mne.preprocessing.annotate_flat helps us to mark flat segments (and channels)
- mne.preprocessing.annotate_muscle_zscore helps us mark segments with muscle activity

We let these functions mostly operate with their default values, and double check the
results.
"""
# %%
# Imports
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_EXTERNAL, OVERWRITE_MSG
from utils import parse_overwrite, prepare_raw_from_source

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

# overwrite existing annotation data?
overwrite = False

# interactive mode: if False, expects the annotations to be loadable
interactive = True

if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=sub,
        data_dir=data_dir,
        analysis_dir=data_dir,
        overwrite=overwrite,
        interactive=interactive,
    )

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    data_dir = defaults["data_dir"]
    analysis_dir = defaults["analysis_dir"]
    overwrite = defaults["overwrite"]
    interactive = defaults["interactive"]

# %%
# Prepare file paths
savedir = analysis_dir / "derived_data" / "annotations"
savedir.mkdir(parents=True, exist_ok=True)

fname_pyprep = savedir / f"sub-{sub:02}_bads_pyprep.json"
fname_bad_channels = savedir / f"sub-{sub:02}_bad-channels.txt"
fname_annots = savedir / f"sub-{sub:02}_annotations.txt"

derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
derivatives.mkdir(parents=True, exist_ok=True)
fname_fif = derivatives / f"sub-{sub:02}_concat_raw.fif.gz"

# %%
# Check overwrite
if (not overwrite) and interactive:
    for fname in [fname_bad_channels, fname_annots, fname_fif]:
        if fname.exists():
            raise RuntimeError(OVERWRITE_MSG.format(fname))

# %%
# Prepare data
raw = prepare_raw_from_source(sub, data_dir, analysis_dir)

# %%
# Handle original annotations
annots_orig = raw.annotations.copy()

bv_annots_to_remove = [
    "Comment/ControlBox is not connected via USB",
    "New Segment/",
    "Comment/actiCAP Data On",  # only sub-10
]
idxs_to_remove = []
for annot in bv_annots_to_remove:
    idxs = np.nonzero(annots_orig.description == annot)[0].tolist()
    idxs_to_remove += idxs

annots_orig.delete(idxs_to_remove)

# if we run in non-interactive mode, we can exit here
if not interactive:
    if not all((fname_bad_channels.exists(), fname_annots.exists())):
        raise RuntimeError(f"Did not find annotation files for sub-{sub:02}.")

    # set bad segments
    annots_bad = mne.read_annotations(fname_annots)
    annots_all = annots_orig + annots_bad
    raw = raw.set_annotations(annots_all)

    # set bad channels
    prev = raw.info["bads"]
    raw.load_bad_channels(fname_bad_channels)
    print(f"Setting bad channels: {prev} -> {raw.info['bads']}")

    raw.save(fname_fif, overwrite=overwrite)
    sys.exit()

# ... else, we continue with interactive screening of the data
# remove annotations from raw, we'll add some more on a blank slate now
raw = raw.set_annotations(None)

# pre-mark bad channels identified in "01_find_bads.py"
if fname_pyprep.exists():
    with open(fname_pyprep, "r") as fin:
        bads_dict = json.load(fin)
    raw.info["bads"] += bads_dict["bad_all"]
    bads_str = json.dumps(bads_dict, indent=4)
    print(f"Pre-marking bad channels, please double check:\n\n{bads_str}\n\n")
else:
    print("Not pre-marking any bad channels. Consider running '01_find_bads.py'.")

# %%
# Automatically add annotations on bad segments
# mark flat segments
annots_flat, bads = mne.preprocessing.annotate_flat(raw, picks="eeg", min_duration=0.5)
raw.info["bads"] += bads

# Bad "break" segments, temporarily add original annotations back for the algorithm
raw = raw.set_annotations(annots_orig)
annots_break = mne.preprocessing.annotate_break(
    raw,
    min_break_duration=3,
    t_start_after_previous=1,
    t_stop_before_next=1,
    ignore=(
        "bad",
        "edge",
        "Stimulus/S 90",
        "Stimulus/S190",
        "Stimulus/S  8",
        "Stimulus/S108",
    ),
)
raw = raw.set_annotations(None)

# bad muscle segments
raw_copy = raw.copy()
raw_copy.notch_filter([50, 100])

threshold_muscle = 5
annots_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
    raw_copy, threshold=threshold_muscle, ch_type="eeg"
)
annots_muscle = mne.Annotations(
    annots_muscle.onset,
    annots_muscle.duration,
    annots_muscle.description,
    orig_time=raw.info["meas_date"],
)

# delete bad segments that are nested in a bad break segment
for annots_bad_search in [annots_flat, annots_muscle]:
    bad_remove_idxs = []
    for annot_b in annots_break:
        onset = annot_b["onset"]
        offset = onset + annot_b["duration"]
        for i, _annot in enumerate(annots_bad_search):
            if (_annot["onset"] >= onset) and (_annot["onset"] < offset):
                bad_remove_idxs.append(i)

    annots_bad_search.delete(bad_remove_idxs)


# combine all automatically identified bad segments
annots_bad = annots_flat + annots_break + annots_muscle
raw = raw.set_annotations(annots_bad)

# %%
# Inspect the channel-wise power spectrum
# This will open in a separate window, click on individual lines to see channel names
matplotlib.use("Qt5Agg")
with sns.plotting_context("talk"):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axvline(10, color="gray", linestyle="-.")
    fig = raw.plot_psd(picks="eeg", fmax=60, ax=ax)
    sns.despine(fig)
plt.show(block=True)

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
with open(fname_bad_channels, "w") as fout:
    lines = "\n".join(raw.info["bads"])
    fout.writelines(lines)

# Save bad annotations
raw.annotations.save(fname_annots, overwrite=overwrite)

# Save data with all annots as FIF
annots_all = annots_orig + raw.annotations
raw.set_annotations(annots_all)
raw.save(fname_fif, overwrite=overwrite)

# %%
