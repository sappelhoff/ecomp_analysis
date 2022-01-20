"""Calculate RDMs based on Mahalanobis distance.

For a given subject:
- load epochs for a stream (single or dual)
- perform baseline correction
- create a design matrix where each trial is predicted by the condition of that
  trial. For example, if there are conditions 1, 2, and 3, the design matrix
  has three columns and len(epochs) rows, with each row corresponding to an epoch.
  The design matrix will have an entry of 1 in the second column (condition 2) in
  the 12th row, if the 12th epoch was of condition 2. The other entries are 0.
  Thus, the sum over columns is a vector of 1s, of length len(epochs).
- zscore the EEG data (epochs x channels x timepoints) over epochs (=trials).
  This centers the data and allows us to not add a constant term to the design
  matrix.
- Run an ordinary least squares regression for each channel and timepoint to
  predict the amplitude over trials, using the design matrix. As a result we
  will obtain the regression coefficients and the residuals.
- The coefficients will be of shape (conditions x channels x timepoints),
  where each condition coefficient corresponds to the ERP of that condition.
- The residuals will be of the same shape as the EEG data
  (epochs x channels x timepoints). We can use the Ledoit Wolf method to
  compute a covariance matrix (channels x channels) at each timepoint.
- Using the regression coefficients and the covariance matrix, we can
  compute the Mahalanobis distance (over channels) for each timepoint,
  gicing us a representational dissimilarity matrix per timepoint of
  shape (conditions x conditions x timepoints).

"""
# %%
# Imports
import sys
from pathlib import Path

import mne
import numpy as np
import scipy.stats
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
from sklearn.covariance import LedoitWolf
from tqdm.auto import tqdm

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_EXTERNAL
from utils import parse_overwrite

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

# overwrite existing data?
overwrite = False

# which baseline to apply
baseline = (None, 0)

# other settings
numbers = range(1, 10)
streams = ["single", "dual"]

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=sub,
        data_dir=data_dir,
        analysis_dir=analysis_dir,
        overwrite=overwrite,
    )

    defaults = parse_overwrite(defaults)

    sub = defaults["sub"]
    data_dir = defaults["data_dir"]
    analysis_dir = defaults["analysis_dir"]
    overwrite = defaults["overwrite"]

if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

# %%
# Prepare file paths
derivatives = data_dir / "derivatives"

fname_epo = derivatives / f"sub-{sub:02}" / f"sub-{sub:02}_numbers_epo.fif.gz"

mahal_dir = data_dir / "derivatives" / "rsa" / "rdms_mahalanobis"
mahal_dir.mkdir(exist_ok=True, parents=True)

fname_rdm_template = mahal_dir / "sub-{:02}_stream-{}_rdm-mahal.npy"

# %%
# Calculate RDMs for each stream
for stream in streams:
    # Read data
    epochs = mne.read_epochs(fname_epo, preload=False, verbose=False)
    epochs = epochs[stream]
    epochs.load_data()
    ntrials = len(epochs)

    # apply baseline
    epochs.apply_baseline(baseline)

    # Prepare a design matrix
    ntrials = len(epochs)
    conditions = np.unique(epochs.metadata["number"])
    nconditions = len(conditions)
    design_matrix = np.zeros((ntrials, nconditions))

    for icondi, condi in enumerate(conditions):

        assert isinstance(condi, np.int64)
        idxs = epochs.metadata["number"] == condi

        design_matrix[idxs, icondi] = 1

    # sanity check that sum of each row is 1
    assert np.sum(design_matrix.sum(axis=1) == 1) == len(epochs)

    # Run OLS for each channel-time bin over epochs
    # first zscore the data
    eeg_data_V = epochs.get_data()
    _, nchs, ntimes = eeg_data_V.shape
    assert _ == ntrials
    eeg_data_V = scipy.stats.zscore(eeg_data_V, axis=0)

    coefs = np.zeros((nconditions, nchs, ntimes))
    resids = np.zeros_like(eeg_data_V)
    for ichannel in tqdm(range(nchs)):
        for itime in range(ntimes):
            y = eeg_data_V[..., ichannel, itime]

            model = sm.OLS(endog=y, exog=design_matrix, missing="raise")
            results = model.fit()

            coefs[..., ichannel, itime] = results.params
            resids[..., ichannel, itime] = results.resid

    # Calculate pairwise mahalanobis distance between regression coefficients
    rdm_times = np.zeros((nconditions, nconditions, ntimes))
    for itime in tqdm(range(ntimes)):
        response = coefs[..., itime]
        residuals = resids[..., itime]

        # Estimate covariance from residuals
        lw_shrinkage = LedoitWolf(assume_centered=True)
        cov = lw_shrinkage.fit(residuals)  # see cov.covariance_

        # Compute pairwise mahalanobis distances
        VI = np.linalg.inv(cov.covariance_)
        rdm = squareform(pdist(response, metric="mahalanobis", VI=VI))
        assert ~np.isnan(rdm).all()
        rdm_times[..., itime] = rdm

    # Sanity check: compare coef to ERP --> must be the same
    # need to zscore epochs first
    epochs._data = scipy.stats.zscore(epochs._data, axis=0)
    erps = np.zeros_like(coefs)
    for inumber, number in enumerate(numbers):
        erps[inumber, ...] = epochs[f"{number}"].average().data

    np.testing.assert_allclose(coefs, erps)

    # Save RDMs
    fname = str(fname_rdm_template).format(sub, stream)
    if Path(fname).exists() and not overwrite:
        raise RuntimeError(f"File exists and overwrite is False:\n {fname}")
    np.save(fname, rdm_times)
