"""Perform epoching.

Here we make epochs out of the cleaned raw data.
Our main target are epochs centered on each sample number (6000 per subj)
--> "number epochs".

However, for sanity checking the EEG data, we will also create epochs centered
on the button press for each choice (600 per subj). We can then use these
"response epochs" for two simple checks: contrasting left/right button presses,
and (100ms after a button press) correct/wrong feedback.

Note that all settings below are only for the "number epochs". We hardcode the
settings for the "response epochs", because they are used for quick sanity checks
only.

How to use the script?
----------------------
Either run in an interactive IPython session and have code cells rendered ("# %%")
by an editor such as VSCode, **or** run this from the command line, optionally
specifying settings as command line arguments:

```shell

python 05_make_epochs.py --sub=1

```

"""
# %%
# Imports
import sys

import mne
import numpy as np
import pandas as pd
from mne_faster import find_bad_epochs

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL
from utils import get_first_task, get_sourcedata, parse_overwrite

# %%
# Settings
sub = 1

data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

downsample_freq = 250

overwrite = False

t_min_max_epochs = (-0.1, 0.9)

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=sub,
        data_dir=data_dir,
        analysis_dir=data_dir,
        overwrite=overwrite,
        downsample_freq=downsample_freq,
        t_min_max_epochs=t_min_max_epochs,
    )

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    data_dir = defaults["data_dir"]
    analysis_dir = defaults["analysis_dir"]
    overwrite = defaults["overwrite"]
    downsample_freq = defaults["downsample_freq"]
    t_min_max_epochs = defaults["t_min_max_epochs"]

# %%
# Prepare file paths
derivatives = data_dir / "derivatives" / f"sub-{sub:02}"
fname_fif_clean = derivatives / f"sub-{sub:02}_clean_raw.fif.gz"

fname_epochs_numbers = derivatives / f"sub-{sub:02}_numbers_epo.fif.gz"

fname_epochs_responses = derivatives / f"sub-{sub:02}_responses_epo.fif.gz"

dropped_epochs_data = analysis_dir / "derived_data" / "dropped_epochs.tsv"

participants_tsv = analysis_dir / "derived_data" / "participants.tsv"

# %%
# Read behavioral data to use as metadata
dfs = []
for stream in ["single", "dual"]:
    _, tsv = get_sourcedata(sub, stream, data_dir)
    dfs += [pd.read_csv(tsv, sep="\t")]

# put in order as recorded
first_task = get_first_task(sub, analysis_dir)
if first_task == "dual":
    dfs = dfs[::-1]

# concatenate
df = pd.concat(dfs)

# %%
# Work on metadata format: melt "one row per trial" to "one row per sample"

id_vars = [
    "trial",
    "direction",
    "choice",
    "ambiguous",
    "rt",
    "validity",
    "iti",
    "correct",
    "stream",
    "state",
]
value_vars = [
    "sample1",
    "sample2",
    "sample3",
    "sample4",
    "sample5",
    "sample6",
    "sample7",
    "sample8",
    "sample9",
    "sample10",
]

metadata = df.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name="nth_sample",
    value_name="number",
    ignore_index=False,
)
metadata["nth_sample"] = metadata["nth_sample"].str.lstrip("sample").astype(int)
stream_sort_ascending = False if first_task == "single" else True
metadata = metadata.sort_values(
    by=["stream", "trial", "nth_sample"], ascending=[stream_sort_ascending, True, True]
).reset_index(drop=True)

metadata["color"] = np.sign(metadata["number"]).map({-1: "red", 1: "blue"})
metadata["number"] = np.abs(metadata["number"])

metadata
# %%
# Read raw
raw = mne.io.read_raw_fif(fname_fif_clean, preload=True)

# %%
# Get events: pick only number outcomes
# see ecomp_experiment/define_ttl.py
# 11-19 -> blue digit  1 to 9 "single stream"
# 21-29 -> red digit  1 to 9 "single stream"
# 111-119 -> blue digit  1 to 9 "dual stream"
# 121-129 -> red digit  1 to 9 "dual stream"
event_id = {
    **{f"Stimulus/S{i:>3}": i for i in range(11, 20)},
    **{f"Stimulus/S{i:>3}": i for i in range(21, 30)},
    **{f"Stimulus/S{i:>3}": i for i in range(111, 120)},
    **{f"Stimulus/S{i:>3}": i for i in range(121, 130)},
}
events, event_id = mne.events_from_annotations(raw, event_id=event_id)
assert events.shape[0] == metadata.shape[0]
assert events.shape[0] == 6000  # 2 stream, 300 trials, 10 samples

# Make event_id human readable
event_id_human = {}
for key, val in event_id.items():
    digit = int(str(val)[-1])
    if val < 21:
        new_key = f"single/blue/{digit}"
    elif val < 111:
        new_key = f"single/red/{digit}"
    elif val < 121:
        new_key = f"dual/blue/{digit}"
    else:
        assert val < 130
        new_key = f"dual/red/{digit}"
    event_id_human[new_key] = val
assert len(event_id_human) == 2 * 2 * 9  # 2 streams, 2 colors, 9 digits

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
    metadata=metadata,
    decim=decim,  # based on downsample_freq
    preload=True,
    tmin=tmin,
    tmax=tmax,
    baseline=None,  # baseline can be applied at later points
    picks=["eeg"],  # we won't need the EOG and ECG channels from here on
    reject_by_annotation=True,
)

# %%
# drop epochs automatically according to FASTER pipeline, step 2
bad_epos = find_bad_epochs(epochs)
epochs.drop(bad_epos, reason="FASTER_AUTOMATIC")

# %%
# Save amount of dropped epochs
kept_epos = np.array([False if len(i) > 0 else True for i in epochs.drop_log])
perc_rejected = 100 * (1 - (kept_epos.sum() / len(epochs.drop_log)))
nkept_epos = kept_epos.sum()
data = dict(sub=[sub], nkept_epos=[nkept_epos], perc_rejected=[perc_rejected])
df_epochs = pd.DataFrame.from_dict(data)

# attach to previous data (if it exists)
if dropped_epochs_data.exists():
    df_epochs_data = pd.read_csv(dropped_epochs_data, sep="\t")
    df_epochs = pd.concat([df_epochs_data, df_epochs]).reset_index(drop=True)
    df_epochs = df_epochs_data.sort_values(by="sub")

df_epochs.to_csv(dropped_epochs_data, sep="\t", na_rep="n/a", index=False)
# %%
# save the epochs
epochs.save(fname_epochs_numbers, overwrite=overwrite)

# %%
# Make "response epochs" (see main docstring)
# The code below is "hardcoded", settings above will have limited effect

t_min_max_epochs_responses = (-0.7, 0.8)

df_participants = pd.read_csv(participants_tsv, sep="\t")
ntimeouts = df_participants.loc[
    df_participants["participant_id"] == f"sub-{sub:02}", "ntimeouts"
].to_list()[0]

event_id_responses = {
    **{f"Stimulus/S{i:>3}": i for i in range(31, 35)},
    **{f"Stimulus/S{i:>3}": i for i in range(131, 135)},
}
events_responses, event_id_responses = mne.events_from_annotations(
    raw, event_id=event_id_responses
)
assert events_responses.shape[0] == 600 - ntimeouts  # 2 stream, 300 trials, 1 choice

event_id_human_responses = {}
for key, val in event_id_responses.items():
    if val < 100:
        if val == 31:
            act = "lower"
        else:
            assert val == 32
            act = "higher"
        new_key = f"single/{act}"
    else:
        assert val > 100
        if val == 133:
            act = "blue"
        else:
            assert val == 134
            act = "red"
        new_key = f"dual/{act}"
    event_id_human_responses[new_key] = val
assert len(event_id_human_responses) == 2 * 2  # 2 streams, binary choices

metadata_responses = df[df["validity"]]
assert metadata_responses.shape[0] == 600 - ntimeouts

epochs_responses = mne.Epochs(
    raw=raw,
    events=events_responses,
    event_id=event_id_human_responses,
    metadata=metadata_responses,
    decim=decim,  # based on downsample_freq
    preload=True,
    tmin=t_min_max_epochs_responses[0],
    tmax=t_min_max_epochs_responses[1],
    baseline=None,  # baseline can be applied at later points
    picks=["eeg"],  # we won't need the EOG and ECG channels from here on
    reject_by_annotation=True,
)

bad_epos_responses = find_bad_epochs(epochs_responses)
epochs_responses.drop(bad_epos_responses, reason="FASTER_AUTOMATIC")

epochs_responses.save(fname_epochs_responses, overwrite=overwrite)

# %%
