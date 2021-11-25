"""Miscellaneous analyses of behavior data."""
# %%
# Imports
import pandas as pd

from config import DATA_DIR_LOCAL, get_sourcedata

# %%
# Check how often participants "timed out" on their choices
for sub in range(1, 33):
    for stream in ["single", "dual"]:
        _, tsv = get_sourcedata(sub, stream, DATA_DIR_LOCAL)
        df = pd.read_csv(tsv, sep="\t")
        validity_sum = df["validity"].to_numpy().sum()
        if validity_sum != df.shape[0]:
            print(sub, stream, f" - n timeouts: {df.shape[0]-validity_sum}")
