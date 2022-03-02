"""Functions for cluster based permutation tesing of RSA timecourses."""
import itertools
import operator

import numpy as np
import pandas as pd

from config import NUMBERS, STREAMS, SUBJS


def prep_for_clusterperm(df, model, orth):
    """Prepare data for cluster based permutation testing.

    Parameters
    ----------
    df : pd.DataFrame
        The RSA data for all subjects and streams.
    model : str
        Which model to run tests on.
    orth : bool
        Whether to run on the orthogonalized model or not.

    Returns
    -------
    X : np.ndarray, shape(nsubjs, ntimes, nstreams)
        The data.
    """
    times = np.unique(df["time"])
    X = np.full((len(SUBJS), len(times), len(STREAMS)), np.nan)
    for istream, stream in enumerate(STREAMS):
        df_subset = df[
            (df["model"] == model) & (df["orth"] == orth) & (df["stream"] == stream)
        ]

        # sanity check rows are in expected order
        pd.testing.assert_frame_equal(
            df_subset, df_subset.sort_values(by=["subject", "itime"])
        )

        # reshape similarities to matrix: subject X timepoints
        X[..., istream] = (
            df_subset["similarity"].to_numpy().reshape(len(SUBJS), len(times))
        )

    assert not np.isnan(X).any()
    return X


def perm_X_1samp(X, rng):
    """Permute X for a 1 sample t-test against zero.

    We randomly flip the signs within each participant.
    """
    nsubjs = X.shape[0]
    flip = rng.choice(np.array([-1, 1]), size=nsubjs, replace=True)
    X_perm = np.full_like(X, np.nan)
    for i in range(nsubjs):
        X_perm[i, ...] = X[i, ...] * flip[i]
    return X_perm


def perm_df_anovarm(df, rng):
    """Permute a "value" column in `df` for a repeated measures anova.

    We randomly change the number labels for each subject and provide
    the permuted data in a "value_perm" column.
    """
    # create the subject specific permutation indices
    idxs = {sub: rng.permutation(len(NUMBERS)) for sub in SUBJS}

    # Go through the data and re-index the values
    perm_idx = np.full(len(df), np.nan)
    count = 0
    for stream, grp_stream in df.groupby("stream"):
        for subj, grp_subj in grp_stream.groupby("subject"):
            for time, grp_time in grp_subj.groupby("time"):
                # down at this level, the grp_time dataframe has 9 rows,
                # one for each number (ordered 1-9).
                # Shuffle the indices using our pre-generated permutation
                # index for this subject
                perm_idx[count : count + len(NUMBERS)] = (
                    grp_time.index.to_numpy()[idxs[subj]]
                ).tolist()
                count += len(NUMBERS)
    df["value_perm"] = df.loc[perm_idx, "value"].to_numpy()
    return df


def return_clusters(arr):
    """Return a list of clusters (=list of indices).

    Parameters
    ----------
    arr : np.ndarray, shape(n,)
        A one dimensional array of zeros and ones, each index reflecting a
        timepoint. When the value is one, the timepoint is significant.

    Returns
    -------
    clusters : list of lists
        Each entry in the list is a list of indices into time. Each of these
        indices marks a significant timepoint.
    """
    # See: https://stackoverflow.com/a/31544396/5201771
    return [
        [i for i, value in it]
        for key, it in itertools.groupby(enumerate(arr), key=operator.itemgetter(1))
        if key != 0
    ]


def get_max_stat(clusters, stat="length", tvals=None):
    """Get maximum cluster statistic.

    Parameters
    ----------
    clusters : list of lists
        Each entry in the list is a list of indices into time.
    stat : {"length", "mass"}
        Whether to return cluster length or cluster mass
        statistic. ``"mass"`` requires `tvals` to be specified
    tvals : np.ndarray
        The t-values used for computing cluster mass if `stat`
        is "mass".

    Returns
    -------
    maxstat : float
        The maximum cluster statistic.
    """
    maxstat = 0
    cluster_lengths = [len(cluster) for cluster in clusters]
    if len(cluster_lengths) == 0:
        # if no clusters, return early
        return maxstat
    if stat == "length":
        maxstat = np.max(cluster_lengths)
    else:
        assert stat == "mass"
        cluster_masses = [np.sum(np.abs(tvals[cluster])) for cluster in clusters]
        maxstat = np.max(cluster_masses)
    return maxstat


def get_significance(distr, stat, clusters_obs, tvals_obs, clusterthresh):
    """Evaluate significance of observed clusters.

    Parameters
    ----------
    distr : np.ndarray, shape(n,)
        The permutation distribution.
    stat : {"length", "mass"}
        Whether to return cluster length or cluster mass statistic.
        ``"mass"`` requires `tvals` to be specified
    clusters_obs : list of lists
        The observed clusters. The nested lists contain time indices.
    tvals_obs : np.ndarray
        The observed t-values used for computing cluster mass if `stat` is "mass"
    clusterthresh : float
        The threshold to consider for determining significance of a cluster.

    Returns
    -------
    clusterthresh_stat : int
        The cluster statistic cutoff. Statistics higher than this are
        considered significant.
    cluster_stats : list of float
        The observed cluster statistics.
    cluster_stats_obs : list of lists
        The observed clusters that are significant at `clusterthresh`.
    pvals : list of float
        The p-values associated with the significant clusters.
    """

    def _calc_pval(distr, clu_stat):
        return (1 + np.sum(distr >= clu_stat)) / (1 + len(distr))

    # Calculate significance "threshold" in terms of sampled statistics
    clusterthresh_stat = 0
    step = 1 if stat == "length" else 0.01
    while True:
        pval = _calc_pval(distr, clusterthresh_stat)
        if (pval < clusterthresh) or (pval == 1 / (1 + len(distr))):
            break
        clusterthresh_stat += step

    # Find significant observed clusters
    is_significant = []
    pvals = []
    if stat == "length":
        cluster_stats_obs = [len(cluster) for cluster in clusters_obs]
    else:
        assert stat == "mass"
        cluster_stats_obs = [
            np.sum(np.abs(tvals_obs[cluster])) for cluster in clusters_obs
        ]
    for clu_stat in cluster_stats_obs:
        pval = _calc_pval(distr, clu_stat)
        is_significant.append(pval < clusterthresh)
        pvals.append(pval)

    sig_clusters = [clu for clu, sig in zip(clusters_obs, is_significant) if sig]
    pvals = [pval for pval, sig in zip(pvals, is_significant) if sig]
    return clusterthresh_stat, cluster_stats_obs, sig_clusters, pvals
