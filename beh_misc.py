"""Miscellaneous analyses of behavior data."""
# %%
# Imports
import json

import pandas as pd

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_LOCAL
from utils import get_sourcedata

# %%
# Settings

analysis_dir = ANALYSIS_DIR_LOCAL
data_dir = DATA_DIR_LOCAL

# %%
# Check how often participants "timed out" on their choices
for sub in range(1, 33):
    for stream in ["single", "dual"]:
        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        validity_sum = df["validity"].to_numpy().sum()
        if validity_sum != df.shape[0]:
            print(sub, stream, f" - n timeouts: {df.shape[0]-validity_sum}")

# %%
# Create a BIDS style participants.tsv file
keys = ["participant_id", "age", "sex", "handedness", "first_task", "included"]
data = {key: [] for key in keys}
for sub in range(1, 33):
    infos = []
    for stream in ["single", "dual"]:

        fname = (
            data_dir
            / "sourcedata"
            / f"sub-{sub:02}"
            / f"sub-{sub:02}_stream-{stream}_info.json"
        )

        with open(fname, "r") as fin:
            infos += [json.load(fin)]

    # sanity check that keys are identical as expected
    for key in ["Age", "Handedness", "ID", "Sex", "experiment_version"]:
        assert (
            infos[0][key] == infos[1][key]
        ), f"info mismatch for sub-{sub}, key '{key}'"

    # experiment_version should have been the same for every participant
    assert infos[0]["experiment_version"] == "2021.1.0"

    # stream and recording_datetime should differ
    assert infos[0]["stream"] != infos[1]["stream"]
    assert infos[0]["recording_datetime"] != infos[1]["recording_datetime"]

    if infos[0]["recording_datetime"] < infos[1]["recording_datetime"]:
        first_task = infos[0]["stream"]
    else:
        first_task = infos[1]["stream"]
    first_task

    age = infos[0]["Age"]
    sex = infos[0]["Sex"].lower()
    handedness = infos[0]["Handedness"].lower()
    included = sub not in BAD_SUBJS

    for key, val in zip(
        keys, [f"sub-{sub:02}", age, sex, handedness, first_task, included]
    ):
        data[key] += [val]

# save
participants_tsv = pd.DataFrame.from_dict(data)
fname = analysis_dir / "derived_data" / "participants.tsv"
participants_tsv.to_csv(fname, sep="\t", na_rep="n/a", index=False)

# print descriptives for included participants
partici_df = participants_tsv[participants_tsv["included"]]
print(partici_df["sex"].value_counts(), end="\n\n")
print(partici_df["handedness"].value_counts(), end="\n\n")
print(partici_df["age"].describe().round(2), end="\n\n")


# %%
