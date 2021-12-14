"""Miscellaneous analyses of EEG data."""
# %%
# Imports
import numpy as np
import pandas as pd

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS

# %%
# Settings

analysis_dir = ANALYSIS_DIR_LOCAL

# %%
# File paths
dropped_epochs = analysis_dir / "derived_data" / "dropped_epochs.tsv"

bad_channels_template = (
    analysis_dir / "derived_data" / "annotations" / "sub-{:02}_bad-channels.txt"
)

# %%
# Analyze dropped epochs
df_dropped_epochs = pd.read_csv(dropped_epochs, sep="\t")

mean_retained = 100 - np.round(df_dropped_epochs["perc_rejected"].mean(), 1)
print(f"On average, {mean_retained} percent of epochs were retained per participant.")

# %%
# Analyze interpolated channels
data = dict(sub=[], nbad=[])
for sub in range(1, 33):
    if sub in BAD_SUBJS:
        continue
    fname = str(bad_channels_template).format(sub)
    with open(fname, "r") as fin:
        lines = fin.readlines()
    nbad = len(lines)
    data["sub"].append(sub)
    data["nbad"].append(nbad)

df_bad_chs = pd.DataFrame.from_dict(data)

mean_bad = np.round(df_bad_chs["nbad"].mean(), 1)
std_bad = np.round(df_bad_chs["nbad"].std(), 1)

print(f"On average, {mean_bad}+-{std_bad} channels were interpolated per participant.")

# %%
