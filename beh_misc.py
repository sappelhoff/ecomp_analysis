"""Miscellaneous analyses of behavior data."""
# %%
# Imports
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_LOCAL, STREAMS, SUBJS
from utils import get_sourcedata

# %%
# Settings

analysis_dir = ANALYSIS_DIR_LOCAL
data_dir = DATA_DIR_LOCAL

# %%
# Check how often participants "timed out" on their choices
timeout_data = dict(sub=[], ntimeouts=[])
for sub in range(1, 33):
    ntimeouts = 0
    for stream in STREAMS:
        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        validity_sum = df["validity"].to_numpy().sum()
        ntimeouts_stream = df.shape[0] - validity_sum
        ntimeouts += ntimeouts_stream
        if ntimeouts_stream > 0:
            print(sub, stream, f" - n timeouts: {ntimeouts_stream}")
    timeout_data["sub"] += [sub]
    timeout_data["ntimeouts"] += [ntimeouts]

# %%
# Create a BIDS style participants.tsv file
keys = [
    "participant_id",
    "age",
    "sex",
    "handedness",
    "first_task",
    "included",
    "ntimeouts",
]
data = {key: [] for key in keys}
for sub in range(1, 33):

    infos = []
    for stream in STREAMS:

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
    ntimeouts = timeout_data["ntimeouts"][timeout_data["sub"].index(sub)]

    for key, val in zip(
        keys, [f"sub-{sub:02}", age, sex, handedness, first_task, included, ntimeouts]
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
# Print average number of timeouts per participant
_ = np.round(np.mean((partici_df["ntimeouts"] / 600) * 100), 2)
print(f"On average there were {_}% timouts per participant.")

# %%
# Compute Binomial tests for bad subjects
for sub in BAD_SUBJS:
    for stream in STREAMS:
        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")

        # binomial test on performance)
        n = df[(~df["ambiguous"]) & (df["validity"])]["correct"].to_numpy().shape[0]
        successes = (
            df[(~df["ambiguous"]) & (df["validity"])]["correct"].to_numpy().sum()
        )
        res = binomtest(k=successes, n=n, p=0.5, alternative="two-sided")
        print(sub, stream, res, "\n")

# %%
# Response time for subjects
dfs = []
for sub in SUBJS:
    for stream in STREAMS:
        _, tsv = get_sourcedata(sub, stream, data_dir)
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "subject", sub)
        dfs.append(df)

df = pd.concat(dfs).reset_index(drop=True)
assert len(df) == len(SUBJS) * len(STREAMS) * 300

# Print summary
print(
    "Response times by stream:\n\n",
    df.groupby("stream")["rt"].describe().round(2),
    "\n\nResponse times overall:\n\n",
    df["rt"].describe().round(2),
)

# plot
with sns.plotting_context("talk"):

    # distributions
    sns.displot(x="rt", data=df, hue="stream")

    # means
    fig, ax = plt.subplots()
    sns.pointplot(x="stream", y="rt", data=df, ci=68, ax=ax)
    sns.despine(fig)


# %%
