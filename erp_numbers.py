"""Compute and plot ERPs over subjects.

TODO:
- speed up
- potentially save intermediate data (and track in git)
- try with other P3 group (see NHB2017 instead of Appelhoff2021)
- use timewindow based on some stats?

"""
# %%
# Imports

import matplotlib.pyplot as plt
import mne
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL, SUBJS

# %%
# Settings
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

derivatives = data_dir / "derivatives"

subjects = SUBJS
baseline = (None, 0)

streams = ["single", "dual"]
numbers = range(1, 10)

# %%
# Helper funcs


def get_erp(fname_epo, sel=None):
    """Load epochs and return average."""
    epochs = mne.read_epochs(fname_epo, preload=True, verbose=0)
    if sel is None:
        return epochs.average()
    return epochs[sel].average()


def get_cocktail_blanks(subjects, derivatives, baseline, sel=None):
    """Get cocktail blank for a selection."""
    cocktail_blanks = {}
    for sub in tqdm(subjects):
        fname_epo = derivatives / f"sub-{sub:02}" / f"sub-{sub:02}_numbers_epo.fif.gz"
        erp = get_erp(fname_epo, sel)
        erp.apply_baseline(baseline)
        cocktail_blanks[sub] = erp.data
    return cocktail_blanks


def get_dict_epochs(subjects, derivatives):
    """Get non-preloaded, non-baselined epochs."""
    dict_epochs = {}
    for sub in tqdm(subjects):
        fname_epo = derivatives / f"sub-{sub:02}" / f"sub-{sub:02}_numbers_epo.fif.gz"
        epochs = mne.read_epochs(fname_epo, preload=False, verbose=False)
        dict_epochs[sub] = epochs
    return dict_epochs


# %%
# Get data

# cocktail blank ERPs
dict_cocktail_blanks = {}
for stream in streams:
    dict_cocktail_blanks[stream] = get_cocktail_blanks(
        subjects, derivatives, baseline, sel=stream
    )

# non-preloaded epochs
dict_epochs = get_dict_epochs(subjects, derivatives)


# %%

dict_streams = {j: {str(i): [] for i in numbers} for j in streams}
p3_group = ["Cz", "C1", "C2", "CPz", "CP1", "CP2", "CP3", "CP4", "Pz", "P1", "P2"]
mean_times = (0.3, 0.7)
start, stop = list(dict_epochs.values())[0].time_as_index(mean_times)

dict_mean_amps = {j: {str(i): [] for i in numbers} for j in streams}
for stream in tqdm(dict_streams):

    dict_numbers = dict_streams[stream]

    for number in tqdm(dict_numbers):

        # TODO:
        # probably faster to iterate: sub -> stream -> number
        # and load data per subj once for all conditions
        for sub, epo in dict_epochs.items():
            epo_sel = epo[f"{stream}/{number}"]
            epo_sel.load_data()
            epo_sel.apply_baseline(baseline)
            evoked = epo_sel.average()

            # cocktail blank removal
            evoked.data = evoked.data - dict_cocktail_blanks[stream][sub]

            # extract mean amplitude
            with mne.utils.use_log_level(False):
                df_epo = (
                    evoked.to_data_frame(picks=p3_group, long_format=True)
                    .groupby("time")
                    .mean()
                    .reset_index(drop=True)
                )

            # TODO: figure out better way than adding +2 to stop idx
            mean_amp = df_epo.iloc[start : stop + 2]["value"].mean()
            dict_mean_amps[stream][number] += [mean_amp]

            dict_numbers[number] += [evoked]


# %%
# Plot
cmap = sns.color_palette("crest_r", as_cmap=True)
for stream in streams:
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
# Gather mean amplitude data
all_dfs = []
for stream in streams:
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
assert df_mean_amps.shape[0] == len(subjects) * len(numbers) * len(streams)
df_mean_amps
# %%
fig, ax = plt.subplots()
sns.pointplot(x="number", y="mean_amp", hue="stream", data=df_mean_amps, ci=68, ax=ax)
sns.despine(fig)
# %%
