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

- For each subject:
    - Load data from both tasks and concatenate to a single raw file
    - anonymize the file (see below)
    - Set channel types (EEG, EOG, ECG) and add electrode positions (standard montage)
    - Get all annotations, and remove the BrainVision ones ("New Segment", ...)
    - Save these annotations in a variable ("annots_orig")
    - Remove existing annotations from raw, for browsing the data without them
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

python 01_raw_inspection.py --sub=1

```

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

- mne.preprocessing.annotate_break to mark block breaks
- mne.preprocessing.annotate_flat helps us to mark flat segments (and channels)
- mne.preprocessing.annotate_muscle_zscore helps us mark segments with muscle activity

We let these functions mostly operate with their default values, and double check the
results.
"""
# %%
# Imports
import pathlib
import sys

import click
import mne
import numpy as np
import pandas as pd

from config import (
    ANALYSIS_DIR,
    BAD_SUBJS,
    DATA_DIR_EXTERNAL,
    get_daysback,
    get_sourcedata,
)

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source here
data_dir = DATA_DIR_EXTERNAL

# overwrite existing annotation data?
overwrite = False

# interactive mode: if False, expects the annotations to be loadable
interactive = True

# %%


# Potentially overwrite settings with command line arguments
@click.command()
@click.option("-s", "--sub", default=sub, type=int, help="Subject number")
@click.option("-d", "--data_dir", default=data_dir, type=str, help="Data location")
@click.option("-o", "--overwrite", default=overwrite, type=bool, help="Overwrite?")
@click.option("-i", "--interactive", default=interactive, type=bool, help="Interative?")
def get_inputs(sub, data_dir, overwrite, interactive):
    """Parse inputs in case script is run from command line."""
    print("Overwriting settings from command line.\nUsing the following settings:")
    for name, opt in [
        ("sub", sub),
        ("data_dir", data_dir),
        ("overwrite", overwrite),
        ("interactive", interactive),
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
savedir = ANALYSIS_DIR / "derived_data" / "annotations"
savedir.mkdir(parents=True, exist_ok=True)

fname_bad_channels = savedir / f"sub-{sub:02}_bad-channels.txt"

fname_annots = savedir / f"sub-{sub:02}_annotations.txt"

derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
derivatives.mkdir(parents=True, exist_ok=True)
fname_fif = derivatives / f"sub-{sub:02}_concat_raw.fif.gz"

overwrite_msg = "\nfile exists and overwrite is False:\n\n>>> {}\n"

# %%
# Load and concatenate data
if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

raws = []
for stream in ["single", "dual"]:
    vhdr, tsv = get_sourcedata(1, stream, data_dir)
    raw_stream = mne.io.read_raw_brainvision(vhdr, preload=True)
    raws.append(raw_stream)

# put in order as recorded
participants_tsv = ANALYSIS_DIR / "derived_data" / "participants.tsv"
df_participants = pd.read_csv(participants_tsv, sep="\t")
first_task = df_participants.loc[
    df_participants["participant_id"] == f"sub-{sub:02}", "first_task"
].to_list()[0]
if first_task == "dual":
    raws = raws[::-1]

# concatenate
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

# if we run in non-interactive mode, we can exit here
if not interactive:
    if not all((fname_bad_channels.exists(), fname_annots.exists())):
        raise RuntimeError(f"Did not find annotation files for sub-{sub:02}.")

    annots_bad = mne.read_annotations(fname_annots)
    raw = raw.load_bads(fname_bad_channels)
    annots_all = annots_orig + annots_bad
    raw = raw.set_annotations(annots_all)

    if not fname_fif.exists() or overwrite:
        raw.save(fname_fif)
    else:
        raise RuntimeError(overwrite_msg.format(fname_fif))
    sys.exit()

# ... else, we continue with interactive screening of the data
# remove annotations from raw, we'll add some more on a blank slate now
raw = raw.set_annotations(None)

# %%
# Automatically add annotations on bad segments
# mark flat segments
annots_flat, bads = mne.preprocessing.annotate_flat(raw, picks="eeg", min_duration=0.5)
annots_flat = mne.Annotations(
    annots_flat.onset, annots_flat.duration, annots_flat.description
)
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

threshold_muscle = 6
annots_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
    raw_copy, threshold=threshold_muscle, ch_type="eeg"
)

# combine all automatically identified bad segments
annots_bad = annots_flat + annots_break + annots_muscle
raw = raw.set_annotations(annots_bad)

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

# Save bad channels
if not fname_bad_channels.exists() or overwrite:
    with open(fname_bad_channels, "w") as fout:
        lines = "\n".join(raw.info["bads"])
        fout.writelines(lines)
else:
    raise RuntimeError(overwrite_msg.format(fname_bad_channels))

# Save bad annotations
if not fname_annots.exists() or overwrite:
    raw.annotations.save(fname_annots)
else:
    raise RuntimeError(overwrite_msg.format(fname_annots))

# Save data with all annots as FIF
annots_all = annots_orig + raw.annotations
raw.set_annotations(annots_all)

if not fname_fif.exists() or overwrite:
    raw.save(fname_fif)
else:
    raise RuntimeError(overwrite_msg.format(fname_fif))
