"""Sanity check the EEG data."""

# %%
# Imports
import mne
import numpy as np
import pandas as pd

from config import DATA_DIR_EXTERNAL, get_sourcedata

# %%
# Load data
for sub in range(1, 33):
    for stream in ["single", "dual"]:
        print(f"Checking {sub}-{stream}")

        vhdr, tsv = get_sourcedata(sub, stream, DATA_DIR_EXTERNAL)
        with mne.utils.use_log_level(0):
            raw = mne.io.read_raw_brainvision(vhdr)
            events, event_id = mne.events_from_annotations(raw)

        # %%
        # Check amount of triggers are exactly as expected
        events_series = pd.Series(events[:, -1])
        vcounts = events_series.value_counts()

        # Check misc ttl codes, see:
        # https://github.com/sappelhoff/ecomp_experiment/blob/main/ecomp_experiment/define_ttl.py
        occur = {
            80: 1,
            90: 1,
            1: 300,
            2: 300,
            3: 300,
            7: 6,
            8: 6,
        }

        if stream == "dual":
            occur = {key + 100: val for key, val in occur.items()}

        # for explanation of codes 10001 and 99999, see:
        # https://mne.tools/stable/generated/mne.events_from_annotations.html#mne.events_from_annotations
        special = {
            10001: 2,  # e.g., "Comment,ControlBox ...", "Comment,actiCAP Data On", etc.
            99999: 2,  # "New Segment/"
        }

        occur.update(special)

        for value, expected in occur.items():

            # skip this check for a few subjs and specific markers,
            # these are cases where deviations are known and in order
            toskip = {
                "02-dual": "Recording immediately started (not stopped again).",
                "04-single": "Paused twice, instead of once.",
                "10-dual": "Control box was still connected via USB.",
                "19-single": "Recording immediately started (not stopped again).",
            }
            if f"{sub:02}-{stream}" in toskip and value in special:
                continue

            occurrences = vcounts.get(value, 0)
            msg = f"{value} is not as expected: {occurrences} != {expected}"
            assert occurrences == expected, msg

        # %%
        # Check number of digit ttl codes
        # These should be 300 (trials) * 10 (digits per trial)
        digit_vals = list(range(11, 20)) + list(range(21, 30))
        if stream == "dual":
            digit_vals = [val + 100 for val in digit_vals]
        n_occurrences = 0
        for value in digit_vals:
            n_occurrences += vcounts.get(value, 0)
        assert n_occurrences == 3000, n_occurrences

        # %%
        # Check number of choice ttl codes
        # these should be 300 - vcounts.get(30, 0)
        choice_vals = [31, 32, 33, 34]
        n_timeouts = vcounts.get(30, 0)
        if stream == "dual":
            choice_vals = [val + 100 for val in choice_vals]
            n_timeouts = vcounts.get(130, 0)
        n_occurrences = 0
        for value in choice_vals:
            n_occurrences += vcounts.get(value, 0)
        assert n_occurrences == (300 - n_timeouts), n_occurrences

        # %%
        # Check number of feedback ttl codes
        fdbk_timeout = 6 if stream == "single" else 106
        assert n_timeouts == vcounts.get(fdbk_timeout, 0)

        # these should be 300
        fdbk_vals = [4, 5, 6]
        if stream == "dual":
            fdbk_vals = [val + 100 for val in fdbk_vals]
        n_occurrences = 0
        for value in fdbk_vals:
            n_occurrences += vcounts.get(value, 0)
        assert n_occurrences == 300, n_occurrences

        # %%
        # Crosscheck with behavioral data
        df = pd.read_csv(tsv, sep="\t")

        # %%
        # Check that timeouts fit
        assert df["validity"].sum() == (300 - n_timeouts)

        # Make sure not too many are ambiguous (if there are: could be chance)
        assert df["ambiguous"].sum() < 30

        # Inter-trial-interval should be within bounds
        assert df["iti"].min() >= 500
        assert df["iti"].max() <= 1500

        # %%
        # Ensure there is no NaN in the data
        data = raw.get_data()
        assert not np.isnan(data).any()

        print("    done!")
# %%
