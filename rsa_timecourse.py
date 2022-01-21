"""Calculate RSA timecourse."""
# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, DATA_DIR_EXTERNAL, SUBJS
from utils import calc_rdm, rdm2vec

# %%
# Settings
# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

rsa_method = "pearson"
distance_measure = "mahalanobis"

numbers = np.arange(1, 10, dtype=int)
streams = ["single", "dual"]

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"

mahal_dir = data_dir / "derivatives" / "rsa" / "rdms_mahalanobis"
mahal_dir.mkdir(exist_ok=True, parents=True)

fname_rdm_template = str(mahal_dir / "sub-{:02}_stream-{}_rdm-mahal.npy")
fname_times = mahal_dir / "times.npy"

# %%
# Get times for RDM timecourses
times = np.load(fname_times)

# %%
# Calculate model RDM
numberline = calc_rdm(numbers, normalize=True)

# %%
# Calculate RSA per subj and stream
df_rsa_list = []
for sub in tqdm(SUBJS):
    for stream in streams:
        rdm_times = np.load(fname_rdm_template.format(sub, stream))

        # Correlation
        ntimes = rdm_times.shape[-1]
        x = rdm2vec(numberline, lower_tri=True)
        corr_model_times = np.full((ntimes, 1), np.nan)
        for itime in range(ntimes):
            y = rdm2vec(rdm_times[..., itime], lower_tri=True)
            if rsa_method == "pearson":
                corr, _ = scipy.stats.pearsonr(x, y)
            else:
                raise RuntimeError(f"invalid rsa_method: {rsa_method}")
            corr_model_times[itime, 0] = corr

        # Make a dataframe
        _df_rsa = pd.DataFrame(corr_model_times, columns=["similarity"])
        _df_rsa.insert(0, "model", "numberline")
        _df_rsa.insert(0, "method", rsa_method)
        _df_rsa.insert(0, "measure", distance_measure)
        _df_rsa.insert(0, "itime", range(ntimes))
        _df_rsa.insert(0, "time", times)
        _df_rsa.insert(0, "stream", stream)
        _df_rsa.insert(0, "subject", sub)

        # Save
        df_rsa_list.append(_df_rsa)

df_rsa = pd.concat(df_rsa_list)
df_rsa = df_rsa.reset_index(drop=True)
assert len(df_rsa) == ntimes * len(SUBJS) * len(streams)
df_rsa

# %%
# Plot the data

fig, ax = plt.subplots()

sns.lineplot(data=df_rsa, x="time", y="similarity", hue="stream", ci=68, ax=ax)
ax.axhline(0, color="black", lw=0.25, ls="--")
sns.despine(fig)

# %%
