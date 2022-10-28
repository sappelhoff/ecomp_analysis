"""Functions shared between scripts."""

import json
import pathlib
import warnings

import click
import mne
import numpy as np
import pandas as pd
import scipy.special
from numba import njit

from config import CHOICE_MAP, DEFAULT_RNG_SEED, STREAMS


def get_sourcedata(sub, stream, data_dir):
    """Return path to source .vhdr and .tsv file for sub and stream."""
    sub_sourcedata = data_dir / "sourcedata" / f"sub-{sub:02}"
    substream = f"sub-{sub:02}_stream-{stream}"
    vhdr = sub_sourcedata / f"{stream}" / (substream + ".vhdr")
    tsv = sub_sourcedata / f"{stream}" / (substream + "_beh.tsv")
    return vhdr, tsv


def get_estim_params(sub, stream, x0_type, minimize_method, slug, analysis_dir):
    """Get the estimated model parameters."""
    fpath = analysis_dir / "derived_data" / f"estim_params_{minimize_method}{slug}.tsv"
    df = pd.read_csv(fpath, sep="\t")
    datasel = df[
        (df["subject"] == sub) & (df["stream"] == stream) & (df["x0_type"] == x0_type)
    ]
    param_names = ["bias", "kappa", "leakage", "noise"]
    parameters = datasel[param_names].to_numpy().flatten()
    return parameters


def get_daysback(data_dir):
    """Get the 'daysback' for anonymizing this dataset."""
    try:
        with open(data_dir / "code" / "DAYSBACK.json", "r") as fin:
            data = json.load(fin)
    except FileNotFoundError:
        warnings.warn("Did not find DAYSBACK.json. Retuning 0.")
        data = dict(daysback=0)
    return data["daysback"]


def get_first_task(sub, analysis_dir):
    """Get the first task a subject performed."""
    participants_tsv = analysis_dir / "derived_data" / "participants.tsv"
    if not participants_tsv.exists():
        msg = (
            f"File does not exist:\n    > {participants_tsv}\n\n"
            "Please run the 'beh_misc.py' script first"
        )
        raise RuntimeError(msg)
    df_participants = pd.read_csv(participants_tsv, sep="\t")
    first_task = df_participants.loc[
        df_participants["participant_id"] == f"sub-{sub:02}", "first_task"
    ].to_list()[0]

    return first_task


def prepare_raw_from_source(sub, data_dir, analysis_dir):
    """Prepare raw source data as MNE-Python object.

    Does the following:

        - Load single and dual stream raw files
        - Concatenate raw files in correct order
        - Anonymize the concatenated raw object (see Notes)
        - Set channel types: EEG, EOG, ECG
        - Set standard 10-20 montage
        - Set line noise frequency to 50Hz

    Parameters
    ----------
    sub : int
        The subject ID.
    data_dir : pathlib.Path
        Path to the sourcedata.
    analysis_dir : pathlib.Path
        Path to the analysis directory, containing the "derived_data".

    Returns
    -------
    raw : mne.io.Raw
        The raw data.

    Notes
    -----
    To anonymize the file, we change the recording date by subtracting
    a certain number of days, read from a local file called `DAYSBACK.json`.
    If that file is not available on your system, the recording dates will
    remain unchanged. To make sure that the anonymization serves its purpose,
    `DAYSBACK.json` is not shared publicly.
    """
    raws = []
    for stream in STREAMS:
        vhdr, _ = get_sourcedata(sub, stream, data_dir)
        raw_stream = mne.io.read_raw_brainvision(vhdr, preload=True)
        raws.append(raw_stream)

    # put in order as recorded
    first_task = get_first_task(sub, analysis_dir)
    if first_task == "dual":
        raws = raws[::-1]

    # concatenate
    raw = mne.concatenate_raws(raws)

    # Anonymize data
    daysback = get_daysback(data_dir)
    raw = raw.anonymize(daysback=daysback, keep_his=False, verbose=False)

    # Prepare raw object (ch_types, montage, ...)
    # Set the EOG and ECG channels to their type
    raw = raw.set_channel_types({"ECG": "ecg", "HEOG": "eog", "VEOG": "eog"})

    # Set a standard montage for plotting later
    montage = mne.channels.make_standard_montage("easycap-M1")
    raw = raw.set_montage(montage)

    # Add some recording info
    raw.info["line_freq"] = 50
    return raw


@click.command()
@click.option("--sub", type=int, help="Subject number")
@click.option("--data_dir", type=str, help="Data location")
@click.option("--analysis_dir", type=str, help="Analysis dir")
@click.option("--overwrite", default=False, type=bool, help="Overwrite?")
@click.option("--interactive", default=False, type=bool, help="Interative?")
@click.option("--pyprep_rng", default=DEFAULT_RNG_SEED, type=int, help="PyPrep seed")
@click.option("--ica_rng", default=DEFAULT_RNG_SEED, type=int, help="ICA seed")
@click.option("--low_cutoff", type=float, help="low_cutoff")
@click.option("--high_cutoff", type=float, help="high_cutoff")
@click.option("--downsample_freq", type=int, help="downsample_freq")
@click.option("--t_min_max_epochs", type=(float, float), help="t_min_max_epochs")
@click.option("--recompute_faster", default=False, type=bool, help="recompute_faster")
@click.option("--rdm_size", type=str, help="rdm_size")
@click.option("--do_plot", default=True, type=bool, help="do_plot")
@click.option("--fit_scenario", type=str, help="fit_scenario")
@click.option("--fit_position", type=str, help="fit_position")
@click.option("--norm_leak", default=False, type=bool, help="norm_leak")
def get_inputs(
    sub,
    data_dir,
    analysis_dir,
    overwrite,
    interactive,
    pyprep_rng,
    ica_rng,
    low_cutoff,
    high_cutoff,
    downsample_freq,
    t_min_max_epochs,
    recompute_faster,
    rdm_size,
    do_plot,
    fit_scenario,
    fit_position,
    norm_leak,
):
    """Parse inputs in case script is run from command line.

    See Also
    --------
    parse_overwrite
    """
    # strs to pathlib.Path
    data_dir = pathlib.Path(data_dir) if data_dir else None
    analysis_dir = pathlib.Path(analysis_dir) if analysis_dir else None

    # collect all in dict
    inputs = dict(
        sub=sub,
        data_dir=data_dir,
        analysis_dir=analysis_dir,
        overwrite=overwrite,
        interactive=interactive,
        pyprep_rng=pyprep_rng,
        ica_rng=ica_rng,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        downsample_freq=downsample_freq,
        t_min_max_epochs=t_min_max_epochs,
        recompute_faster=recompute_faster,
        rdm_size=rdm_size,
        do_plot=do_plot,
        fit_scenario=fit_scenario,
        fit_position=fit_position,
        norm_leak=norm_leak,
    )

    return inputs


def parse_overwrite(defaults):
    """Parse which variables to overwrite."""
    print("\nParsing command line options...\n")
    inputs = get_inputs.main(standalone_mode=False, default_map=defaults)
    noverwrote = 0
    for key, val in defaults.items():
        if val != inputs[key]:
            print(f"    > Overwriting '{key}': {val} -> {inputs[key]}")
            defaults[key] = inputs[key]
            noverwrote += 1
    if noverwrote > 0:
        print(f"\nOverwrote {noverwrote} variables with command line options.\n")
    else:
        print("Nothing to overwrite, use defaults defined in script.\n")

    print("Using the following configuration:\n")
    for key, val in defaults.items():
        print(f"{key} = {val}")
    print("\n")

    return defaults


@njit
def eq1(X, bias, kappa):
    """Implement equation 1 from Spitzer et al. 2017 [1]_, [2]_.

    Parameters
    ----------
    X : np.ndarray, shape(n,)
        The input values, normalized to range [-1, 1].
    bias : float
        The bias parameter in the range [-1, 1].
    kappa : float
        The kappa parameter in the range [0, 20].

    Returns
    -------
    dv : np.ndarray, shape(n,)
        The subjective decision values, transformed from `X`

    References
    ----------
    .. [1] Spitzer, B., Waschke, L. & Summerfield, C. Selective overweighting of larger
           magnitudes during noisy numerical comparison. Nat Hum Behav 1, 0145 (2017).
           https://doi.org/10.1038/s41562-017-0145
    .. [2] https://github.com/summerfieldlab/spitzer_etal_2017/blob/master/psymodfun.m
    """
    dv = np.sign(X + bias) * (np.abs(X + bias) ** kappa)
    return dv


def eq2(feature_space, bias, kappa, seq_length=10):
    """Implement equation 2 from Spitzer et al. 2017 [1]_, [2]_.

    Parameters
    ----------
    feature_space : np.ndarray, shape(n, m)
        The stimuli over which to compute `gain`, rescaled to the
        range [-1, 1]. This is typically either the full range of
        `m` stimuli encountered over the experiment, in which case
        `n` is ``1``; or the `m` stimuli encountered per trial,
        in which case `n` is the number of trials. For a trial-wise
        feature space in the dual stream task, individual features
        must be sign flipped according to their color category
        (-1: red, +1: blue).
    bias : float
        The bias parameter in the range [-1, 1].
    kappa : float
        The kappa parameter in the range [0, 20].
    seq_length : int
        The length of the sample sequence. Defaults to 10, which was
        the sample sequence length in the eComp experiment.

    Returns
    -------
    gain : np.ndarray, shape(n, seq_length)
        The gain normalization factor, where `n` is ``1`` if gain
        normalization is to be applied over the feature space of the
        entire experiment; or `n` is the number of trials if gain
        normalization is to be applied trial-wise.
        Can be ``None`` if `gnorm` is ``False``.

    References
    ----------
    .. [1] Spitzer, B., Waschke, L. & Summerfield, C. Selective overweighting of larger
           magnitudes during noisy numerical comparison. Nat Hum Behav 1, 0145 (2017).
           https://doi.org/10.1038/s41562-017-0145
    .. [2] https://github.com/summerfieldlab/spitzer_etal_2017/blob/master/psymodfun.m
    """
    gain = np.sum(np.abs(feature_space + bias) ** kappa, axis=1) / np.sum(
        np.abs(feature_space), axis=1
    )
    # This is to conveniently compute dv / gain later on, where dv
    # is of shape (ntrials, seq_length), and for each trial, all samples
    # in that trial sequence need to be divided by gain.
    gain = np.repeat(gain, seq_length).reshape(-1, seq_length)
    return gain


def eq3(dv, category, gain, gnorm, leakage, norm_leak=False, seq_length=10):
    """Implement equation 3b from Spitzer et al. 2017 [1]_, [2]_.

    Parameters
    ----------
    dv : np.ndarray, shape(n,)
        The subjective decision values.
    category : np.ndarray, shape(n,)
        The category each entry in `dv` belonged to. Entries are
        either -1 or 1 (-1: red, +1: blue). For the "single" stream,
        this must be a vector of 1.
    gain : np.ndarray, shape(n, seq_length) | None
        The gain normalization factor, where `n` is ``1`` if gain
        normalization is to be applied over the feature space of the
        entire experiment; or `n` is the number of trials if gain
        normalization is to be applied trial-wise.
        Can be ``None`` if `gnorm` is ``False``.
    gnorm : bool
        Whether or not to apply gain normalization.
    leakage : float
        The leakage parameter in the range [-1, 1], where ``0`` means
        that each sample in the sequence of length `seq_length` receives
        a weight of ``1`` (no recency); and ``1`` means that all samples
        except the last in the sequence receive a weight of ``0``, and
        the last receives a weight of ``1`` (strongest possible recency).
        Values lower than ``0`` indicate primacy (the opposite of recency).
    norm_leak : bool
        Whether or not to normalize the leakage vector resulting from the
        `leakage` parameter. Has no effect when ``leakage==0``, else it
        ensures that the sum of the leakage vector always equals `seq_length`.
    seq_length : int
        The length of the sample sequence. Defaults to 10, which was
        the sample sequence length in the eComp experiment.

    Returns
    -------
    DV : float
        The decision value after gain normalization and leakage.

    References
    ----------
    .. [1] Spitzer, B., Waschke, L. & Summerfield, C. Selective overweighting of larger
           magnitudes during noisy numerical comparison. Nat Hum Behav 1, 0145 (2017).
           https://doi.org/10.1038/s41562-017-0145
    .. [2] https://github.com/summerfieldlab/spitzer_etal_2017/blob/master/psymodfun.m
    """
    if gnorm:
        dv = dv / gain
    dv_flipped = dv * category
    _, ncols = dv.shape
    if ncols == seq_length:
        leakage_term = (1 - leakage) ** (seq_length - np.arange(1, seq_length + 1))
    else:
        # for running by different fit_position args: no leakage
        leakage_term = np.ones(ncols)
    if norm_leak:
        leakage_term = leakage_term * (seq_length / leakage_term.sum())
    DV = np.dot(dv_flipped, leakage_term)
    return DV


def eq4(DV, noise):
    """Implement equation 4 from Spitzer et al. 2017 [1]_, [2]_.

    Parameters
    ----------
    DV : np.ndarray, shape(n,)
        The decision value after gain normalization and leakage.
        One value per trial (`n` trials).
    noise : float
        The noise parameter in the range [0.01, 8].

    Returns
    -------
    CP : np.ndarray, shape(n,)
        The probability to choose 1 instead of 0.
        One value per trial (`n` trials).

    References
    ----------
    .. [1] Spitzer, B., Waschke, L. & Summerfield, C. Selective overweighting of larger
           magnitudes during noisy numerical comparison. Nat Hum Behav 1, 0145 (2017).
           https://doi.org/10.1038/s41562-017-0145
    .. [2] https://github.com/summerfieldlab/spitzer_etal_2017/blob/master/psymodfun.m
    """
    # numerically stable equivalent of: `CP = 1. / (1. + np.exp(-x))`,
    # where `x = DV / noise`
    x = DV / noise
    CP = scipy.special.expit(x)

    return CP


def calc_rdm(vec, normalize):
    """Calculate an RDM based on an input vector.

    Parameters
    ----------
    vec : np.ndarray, shape(n,)
        1-dimensional vector of length `n`, from which
        to calculate an RDM.
    normalize : bool
        Whether or not to normalize the output `rdm` to
        the range [0, 1] through ``rdm /= rdm.max()``.
        Note that this only works work RDMs that already
        have their minimum at 0. The function will raise
        an AssertionError otherwise.

    Returns
    -------
    rdm : np.ndarray, shape(n, n)
        Each entry in the matrix is the pairwise (absolute)
        difference between corresponding entries from the
        input vector (increasing from top to bottom; and
        from left to right).

    """
    arrs = []
    for i in vec:
        arrs.append(np.abs(vec - i))
    rdm = np.stack(arrs, axis=0)
    if normalize:
        if np.all(rdm == 0):
            raise RuntimeError("Cannot normalize if all zeros.")
        rdm = rdm / rdm.max()
        min0 = np.isclose(rdm.min(), 0)
        max1 = np.isclose(rdm.max(), 1)
        assert min0 and max1
    return rdm


def rdm2vec(rdm, lower_tri=True):
    """Get an RDM as a vector.

    Parameters
    ----------
    rdm : np.ndarray, shape(n, n)
        The representational dissimilarity matrix.
    lower_tri : bool
        If ``True``, return only the lower triangle without the
        diagonal as an output. If ``False``, return the full
        RDM as an output.

    Returns
    -------
    vector : np.ndarray, shape(n, )
        Copy of either the full RDM as a vector or the
        lower triangle without diagonal as a vector.
    """
    assert rdm.ndim == 2
    assert rdm.shape[0] == rdm.shape[1]
    rdm = np.asarray(rdm.copy(), dtype=float)
    if lower_tri:
        lower_triangle_idx = np.tril_indices(rdm.shape[0], k=-1)
        vector = rdm[lower_triangle_idx].flatten()
    else:
        vector = rdm.flatten()
    return vector


def prep_to_plot(rdm):
    """Remove upper triangle, including diagonal, from rdm."""
    tri_idx = np.triu_indices(rdm.shape[0])
    tmprdm = rdm.copy()
    tmprdm = tmprdm.astype(float)
    tmprdm[tri_idx] = np.nan
    return tmprdm


def prep_weight_calc(df, nsamples=10):
    """Prepare data for weights calculation.

    Parameters
    ----------
    df : pd.DataFrame
        The stream-specific participant data.
    nsamples : int
        Number of samples in each trial. Defaults to 10,
        as in the eComp experiment.

    Returns
    -------
    weights_df : pd.DataFrame
    stream : {"single", "dual"}
        The stream that this data is on.
    samples : np.ndarray, shape(n,)
        The samples over trials. This array contains values from 1 to 9,
        `nsamples` per trial, so unless trials had to be dropped due to
        no response, `n` is 3000.
    colors : np.ndarray, shape(n,)
        The corresponding category of each sample, coded as 0 (red) or
        1 (blue).
    positions : np.ndarray, shape(n,)
        The position of each sample in its trial from 0 to 9.
    """
    # work on a copy of the data
    weights_df = df.copy()

    stream = np.unique(weights_df["stream"])[0]

    # remove NaN rows
    nan_row_idxs = np.nonzero(~weights_df["validity"].to_numpy())[0]
    for idx in nan_row_idxs:
        assert pd.isna(weights_df.loc[idx, "direction"])
    weights_df = weights_df.drop(nan_row_idxs)

    # prepare data
    isamples = [f"sample{i}" for i in range(1, nsamples + 1)]  # samples 1 to nsamples
    samples_signed = weights_df.loc[:, isamples].to_numpy().flatten()
    samples = np.abs(samples_signed)
    colors = (np.sign(samples_signed) + 1) / 2  # red=0, blue=1
    positions = np.tile(
        np.arange(nsamples, dtype=int), len(weights_df["trial"])
    )  # sample positions
    assert positions.shape == samples.shape

    return weights_df, stream, samples, colors, positions


def prep_model_inputs(df):
    """Extract variables from behavioral data for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The behavioral data for a specific participant and
        task condition (stream).

    Returns
    -------
    X : np.ndarray, shape(n, 10)
        The data. Each row is a trial, each column is an unsigned sample value
        in the range [1, 9] that has been rescaled to the range [-1, 1]. In the
        eComp experiment, there were always 10 samples. The number of trials,
        `n` will be 300 in most cases but can be lower when a participant failed
        to respond for some trials (these trials are then dropped from analysis).
    categories : np.ndarray, shape(n, 10)
        Signed array (-1, +1) of same shape as `X`. For "dual" stream data, the
        sign represents each sample's color category (-1: red, +1: blue).
        For "single" stream data, this is an array of ones, in order to ignore
        the color category.
    y : np.ndarray, shape(n, 1)
        The participant's choices in this task condition (stream). Each entry
        correspondons to a trial and can be ``0`` or ``1``. In single stream
        condition, 0: "lower", 1: "higher". In dual stream condition:
        0: "red", 1: "blue".
    y_true : np.ndarray, shape(n, 1)
        The "objectively correct" choices per per trial. These are in the same
        format as `y`.
    ambiguous : np.ndarray, shape(n, 1)
        Boolean mask on which of the `n` valid trials contained samples that do
        not result in an objectively correct choice. For example in single stream,
        samples have a mean of exactly 5; or in dual stream, the red and blue
        samples have an identical mean.

    See Also
    --------
    config.CHOICE_MAP
    """
    # drop n/a trials (rows)
    idx_no_na = ~df["choice"].isna()

    # get categories
    sample_cols = [f"sample{i}" for i in range(1, 11)]
    X_signed = df.loc[idx_no_na, sample_cols].to_numpy()
    streams = df["stream"].unique()
    assert len(streams) == 1
    if streams[0] == "single":
        categories = np.ones_like(X_signed, dtype=int)
    else:
        assert streams[0] == "dual"
        categories = np.sign(X_signed)

    # Rescale sample values to range [-1, 1]
    # 1, 2, 3, 4, 5, 6, 7, 8, 9 --> -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
    X_abs = np.abs(X_signed)
    X = np.interp(X_abs, (X_abs.min(), X_abs.max()), (-1, +1))
    assert X_abs.min() == 1
    assert X_abs.max() == 9

    # map choices to 0 and 1
    y = df.loc[idx_no_na, "choice"].map(CHOICE_MAP).to_numpy()

    # Get "correct" choices. Single: 0=lower, 1=higher ; Dual: 0=red, 1=blue
    y_true = np.sign(np.sum((X * categories), axis=1)) / 2 + 0.5
    ambiguous = y_true == 0.5

    return X, categories, y, y_true, ambiguous


def psychometric_model(
    parameters, X, categories, y, return_val, gain=None, gnorm=False, norm_leak=False
):
    """Model the behavioral data as in Spitzer 2017, NHB [1]_.

    Parameters
    ----------
    parameters : np.ndarray, shape(4,)
        An array containing the 4 parameters of the model::
            - bias : float
                The bias parameter (`b`) in range [-1, 1].
            - kappa : float
                The kappa parameter (`k`) in range [0, 20].
            -leakage : float
                The leakage parameter (`l`) in range [-1, 1].
            - noise : float
                The noise parameter (`s`) in range [0.01, 8].
    X : np.ndarray, shape(n, 10)
        The data per participant and stream. Each row is a trial, each column
        is an unsigned sample value in the range [1, 9] that has been rescaled
        to the range [-1, 1]. In the eComp experiment, there were always
        10 samples. The number of trials, `n` will be 300 in most cases but can
        be lower when a participant failed to respond for some trials (these
        trials are then dropped from analysis).
    categories : np.ndarray, shape(n, 10)
        Signed array (-1, +1) of same shape as `X`. The sign represents each
        sample's color category (-1: red, +1: blue). For "single" stream data,
        this is an array of ones, in order to ignore the color category.
    y : np.ndarray, shape(n, 1)
        The choices per participant and stream. Each entry is the choice on a
        given trial. Can be ``0`` or ``1``. In single stream condition,
        0: "lower", 1: "higher". In dual stream condition: 0: "red", 1: "blue".
    return_val : {"G", "G_noCP", "sse"}
        Whether to return G-statistic based loss or sum of squared errors.
        ``"G_noCP"`` returns the G-statistic based loss *without*
        the additional `CP` return value.
    gain : np.ndarray, shape(n, 10) | None
        The gain normalization factor, where `n` is ``1`` if gain
        normalization is to be applied over the feature space of the
        entire experiment; or `n` is the number of trials if gain
        normalization is to be applied trial-wise.
        Can be ``None`` if `gnorm` is ``False``.
    gnorm : bool
        Whether to gain-normalize or not. Defaults to ``False``.
    norm_leak : bool
        Whether to normalize the leakage parameter, see ``utils.eq3``.

    Returns
    -------
    loss : float
        To be minimized for parameter estimated. Either based on the G statistic or
        the "sum of squared errors" of the model, depending on `return_val`.
    CP : np.ndarray, shape(n,)
        The probability to choose 1 instead of 0. One value per trial (`n` trials).
        Not returned if `return_val` is ``"G_noCP"``.


    References
    ----------
    .. [1] Spitzer, B., Waschke, L. & Summerfield, C. Selective overweighting of larger
           magnitudes during noisy numerical comparison. Nat Hum Behav 1, 0145 (2017).
           https://doi.org/10.1038/s41562-017-0145

    See Also
    --------
    utils.eq1
    utils.eq2
    utils.eq3
    utils.eq4
    config.CHOICE_MAP
    """
    # unpack parameters
    bias, kappa, leakage, noise = parameters

    # Transform sample values to subjective decision weights
    dv = eq1(X=X, bias=bias, kappa=kappa)

    # Obtain trial level decision variables
    DV = eq3(
        dv=dv,
        category=categories,
        gain=gain,
        gnorm=gnorm,
        leakage=leakage,
        norm_leak=norm_leak,
    )

    # Convert decision variables to choice probabilities
    CP = eq4(DV, noise=noise)

    # Compute "model fit"
    if return_val in ["G", "G_noCP"]:
        # fix floating point issues to avoid log(0) = -inf
        # NOTE: These issues happen mostly when `noise` is very
        #       low (<0.1), and the subject chose an option opposite
        #       to the evidence. For example, noise=0.01, DV=0.5, y=0.
        #
        #       --> In this case, CP would be expit(x), where x is
        #           0.5/0.01 = 50, and expit(50) is 1.0, due to limitations
        #           in floating point precision.
        #       --> Next, when calculating the negative log likelihood
        #           for y == 0, we must do -np.log(1 - CP) and thus arrive
        #           at log(0), which evaluates to -inf.
        #
        #       The same can happen in the opposite case, where e.g.,
        #       noise=0.001, DV=-1.0, y=1.
        #
        #       --> Here, the corresponding expit(-1/0.001) is expit(-1000)
        #           and results in 0.
        #       --> Then, when calculating neg. log. lik. for y == 1, we
        #           must do -log(CP) and thus arrive at log(0) and
        #           -inf again.
        #
        #       We solve this issue by picking the floating point value closest
        #       to 1 (0.999999...) or 0 (0.00...01) instead of actually 1 or 0,
        #       whenever we run into this problem.
        #
        CP[(y == 1) & (CP == 0)] = np.nextafter(0.0, np.inf)
        CP[(y == 0) & (CP == 1)] = np.nextafter(1.0, -np.inf)
        # based on the G statistic (https://en.wikipedia.org/wiki/G-test)
        loss = np.sum(2 * np.log(1.0 / CP[y == 1])) + np.sum(
            2 * np.log(1.0 / (1 - CP[y == 0]))
        )

        if return_val == "G_noCP":
            # for scipy.optimize.minimize, we must return a single float
            return loss
    else:
        assert return_val == "sse"
        loss = np.sum((y - CP) ** 2)

    return loss, CP


def spm_orth(X, opt="pad"):
    """Perform a recursive Gram-Schmidt orthogonalisation of basis functions.

    This function was translated from Matlab to Python using the code from
    the SPM MATLAB toolbox on ``spm_orth.m`` [1]_.

    .. warning:: For arrays of shape(1, m) the results are not equivalent
                 to what the original ``spm_orth.m`` produces.

    Parameters
    ----------
    X : numpy.ndarray, shape(n, m)
        Data to perform the orthogonalization on. Will be performed on the
        columns.
    opt : {"pad", "norm"}, optional
        If ``"norm"``, perform a euclidean normalization according to
        ``spm_en.m`` [2]_. If ``"pad"``, ensure that the output is of the
        same size as the input. Defaults to ``"pad"``.

    Returns
    -------
    X : numpy.ndarray
        The orthogonalized data.

    References
    ----------
    .. [1] https://github.com/spm/spm12/blob/3085dac00ac804adb190a7e82c6ef1/spm_orth.m
    .. [2] https://github.com/spm/spm12/blob/f6948fff302fa4d4b80c9c67bb9ddf/spm_en.m
    """
    assert X.ndim == 2, "This function only operates on 2D numpy arrays."
    n, m = X.shape
    if n == 1:
        raise RuntimeError("Function is unreliable for inputs of shape (1, m).")
    X = X[:, np.any(X, axis=0)]  # drop all "all-zero" columns
    rank_x = np.linalg.matrix_rank(X)

    x = X[:, np.newaxis, 0]
    j = [0]
    for i in range(1, X.shape[-1]):
        D = X[:, np.newaxis, i]
        D = D - np.dot(x, np.dot(np.linalg.pinv(x), D))
        if np.linalg.norm(D, 1) > np.exp(-32):
            x = np.concatenate([x, D], axis=1)
            j.append(i)

        if len(j) == rank_x:
            break

    if opt == "pad":
        # zero padding of null space (if present)
        X = np.zeros((n, m))
        X[:, np.asarray(j)] = x

    elif opt == "norm":
        # Euclidean normalization, based on "spm_en.m", see docstring.
        for i in range(X.shape[-1]):
            if np.any(X[:, i]):
                X[:, i] = X[:, i] / np.sqrt(np.sum(X[:, i] ** 2))

    else:
        # spm_orth.m does "X = x" here. We raise an error, because
        # this option is not documented in spm_orth.m
        # X = x
        raise ValueError("opt must be one of ['pad', 'norm'].")

    return X


def find_dot_idxs(ax, n_expected_dots):
    """Find object in axes corresponding to plotted dots."""
    # see also https://stackoverflow.com/a/63171175/5201771
    children = ax.get_children()
    idxs = []
    for ichild, child in enumerate(children):
        try:
            offsets = child.get_offsets()
        except AttributeError:
            continue

        nrows, ncols = offsets.shape
        if ncols == 2 and nrows == n_expected_dots:
            idxs.append(ichild)

    if len(idxs) == 2:
        return idxs
    else:
        raise RuntimeError("Encountered problems identifying dots in plot.")
