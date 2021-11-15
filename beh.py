"""Beh analysis."""
# %%
import pandas as pd

from config import DATA_DIR_LOCAL, get_sourcedata

# %%
# Read data

sub = 1
stream = "single"
_, tsv = get_sourcedata(sub, stream, DATA_DIR_LOCAL)

# %%
df = pd.read_csv(tsv, sep="\t")
df
# %%
