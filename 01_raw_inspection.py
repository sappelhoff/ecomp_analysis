"""Inspect the raw data."""

# %%
from config import get_sourcedata, get_daysback

import mne


vhdr, tsv = get_sourcedata(1, "single")
raw = mne.io.read_raw_brainvision(vhdr)


# %%
