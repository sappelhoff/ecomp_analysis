"""Inspect the raw data."""

# %%
import mne

from config import get_sourcedata

# %%
# Load and prepare data
vhdr, tsv = get_sourcedata(1, "single")
raw = mne.io.read_raw_brainvision(vhdr, preload=True)

# Set the EOG and ECG channels to their type
raw.set_channel_types({"ECG": "ecg", "HEOG": "eog", "VEOG": "eog"})  # %%

# Set a standard montage for plotting later
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# Temporarily remove existing annotations for faster interactive plotting
raw.set_annotations(None)

# %%
