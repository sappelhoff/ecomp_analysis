"""Sanity check the data."""

# %%
from config import get_sourcedata

import pandas as pd
import mne


# %%
# Load data
for sub in range(1, 8):
    for stream in ["single", "dual"]:
        print(sub, stream)

        vhdr, tsv = get_sourcedata(sub, stream)
        with mne.utils.use_log_level(0):
            raw = mne.io.read_raw_brainvision(vhdr)
            events, event_id = mne.events_from_annotations(raw)

        # %%
        # Check amount of triggers are exactly as expected

        events_series = pd.Series(events[:, -1])
        vcounts = events_series.value_counts()

        # Check misc ttl codes
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

        special = {
            10001: 2,
            99999: 2,
        }

        occur.update(special)

        for value, expected in occur.items():
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

        assert n_timeouts == vcounts.get(6, 0)

        # these should be 300
        fdbk_vals = [4, 5]
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

# %%

# %%

# %%
