"""Compute and plot ERPs over subjects.

TODO:
- use timewindow based on some stats?

"""
# %%
# Imports

import matplotlib.pyplot as plt
import mne
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from config import (
    ANALYSIS_DIR_LOCAL,
    DATA_DIR_EXTERNAL,
    NUMBERS,
    P3_GROUP_CERCOR,
    P3_GROUP_NHB,
    STREAMS,
    SUBJS,
)

# %%
# Settings
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

overwrite = False

subjects = SUBJS
baseline = (None, 0)

mean_times = (0.3, 0.7)

p3_group = [P3_GROUP_CERCOR, P3_GROUP_NHB][0]  # 0 or 1

# %%
# Prepare file paths

derivatives = data_dir / "derivatives"

erp_dir = derivatives / "erps" / f"{baseline}-{mean_times}"
erp_dir.mkdir(exist_ok=True, parents=True)
fname_template_erp = "sub-{:02}_stream-{}_number-{}_ave.fif.gz"


# %%
# Helper functions to get data
def _get_mean_amps(dict_streams, mean_times, p3_group):
    dict_mean_amps = {j: {str(i): [] for i in NUMBERS} for j in STREAMS}
    for stream, dict_numbers in dict_streams.items():
        for number, evoked_list in dict_numbers.items():
            for i, evoked in enumerate(evoked_list):
                start, stop = evoked.time_as_index(mean_times, use_rounding=True)
                # extract mean amplitude and save
                with mne.utils.use_log_level(False):
                    df_epo = (
                        evoked.to_data_frame(picks=p3_group, long_format=True)
                        .groupby("time")
                        .mean()
                        .reset_index(drop=False)
                    )

                assert df_epo.iloc[start]["time"] / 1000 == mean_times[0]
                assert df_epo.iloc[stop]["time"] / 1000 == mean_times[1]
                mean_amp = df_epo.iloc[start : stop + 1]["value"].mean()
                dict_mean_amps[stream][number] += [mean_amp]

    return dict_mean_amps


def _get_erps(sub, baseline):
    # Read data
    fname_epo = derivatives / f"sub-{sub:02}" / f"sub-{sub:02}_numbers_epo.fif.gz"
    epochs = mne.read_epochs(fname_epo, preload=True, verbose=False)

    # apply baseline
    epochs.apply_baseline(baseline)

    # Calculate cocktail blanks per stream
    cocktail_blank = {stream: epochs[stream].average().get_data() for stream in STREAMS}

    # Get cocktailblank corrected ERPs for each stream/number
    # also extract mean amplitudes
    dict_streams = {j: {str(i): [] for i in NUMBERS} for j in STREAMS}
    for stream, dict_numbers in dict_streams.items():
        for number in dict_numbers:

            # form ERP and cocktailblank correct
            sel = f"{stream}/{number}"
            evoked = epochs[sel].average()
            evoked.data = evoked.data - cocktail_blank[stream]

            # add to evokeds
            dict_numbers[number] += [evoked]

    return dict_streams


# %%
# Get subject data
dict_streams = {j: {str(i): [] for i in NUMBERS} for j in STREAMS}
if erp_dir.exists():
    # Try load ERPs from disk directly (faster)
    for stream, dict_numbers in tqdm(dict_streams.items()):
        for number in dict_numbers:
            for sub in SUBJS:
                fname = erp_dir / fname_template_erp.format(sub, stream, number)
                evokeds = mne.read_evokeds(fname, verbose=False)
                assert len(evokeds) == 1
                dict_streams[stream][number] += evokeds
else:
    # Prepare from epochs (slower)
    sub_dicts = {}
    for sub in tqdm(SUBJS):
        sub_dicts[sub] = _get_erps(sub, baseline)

    # Merge subject dicts into one dict of list
    for sub, _dstream in sub_dicts.items():
        for stream, dict_numbers in dict_streams.items():
            for number in dict_numbers:
                dict_streams[stream][number] += _dstream[stream][number]

# Get mean amps
dict_mean_amps = _get_mean_amps(dict_streams, mean_times, p3_group)

# %%
# Plot
cmap = sns.color_palette("crest_r", as_cmap=True)
for stream in STREAMS:
    fig, ax = plt.subplots()
    ax.axvspan(*mean_times, color="black", alpha=0.1)
    with mne.utils.use_log_level(False):
        mne.viz.plot_compare_evokeds(
            dict_streams[stream],
            picks=p3_group,
            combine="mean",
            show_sensors=True,
            cmap=cmap,
            ci=0.68,
            axes=ax,
            title=stream,
        )

# %%
# Gather mean amplitude data into DataFrame
all_dfs = []
for stream in STREAMS:
    df_mean_amps = pd.DataFrame.from_dict(dict_mean_amps[stream])
    assert df_mean_amps.shape[0] == len(subjects)
    df_mean_amps["subject"] = subjects
    df_mean_amps = df_mean_amps.melt(id_vars=["subject"]).sort_values(
        by=["subject", "variable"]
    )
    df_mean_amps = df_mean_amps.rename(
        columns=dict(variable="number", value="mean_amp")
    )
    df_mean_amps.insert(1, "stream", stream)
    all_dfs.append(df_mean_amps)

df_mean_amps = pd.concat(all_dfs)
assert df_mean_amps.shape[0] == len(subjects) * len(NUMBERS) * len(STREAMS)
df_mean_amps

# %%
# Plot mean amps
fig, ax = plt.subplots()
sns.pointplot(x="number", y="mean_amp", hue="stream", data=df_mean_amps, ci=68, ax=ax)
sns.despine(fig)

# %%
# Save ERPs
for stream, dict_numbers in dict_streams.items():
    for number, evoked_list in dict_numbers.items():
        for i, evoked in enumerate(evoked_list):
            sub = SUBJS[i]
            fname = erp_dir / fname_template_erp.format(sub, stream, number)
            if fname.exists() and not overwrite:
                continue
            evoked.save(fname, overwrite)

# %%
