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

from config import DEFAULT_RNG_SEED


def get_sourcedata(sub, stream, data_dir):
    """Return path to source .vhdr and .tsv file for sub and stream."""
    sub_sourcedata = data_dir / "sourcedata" / f"sub-{sub:02}"
    substream = f"sub-{sub:02}_stream-{stream}"
    vhdr = sub_sourcedata / f"{stream}" / (substream + ".vhdr")
    tsv = sub_sourcedata / f"{stream}" / (substream + "_beh.tsv")
    return vhdr, tsv


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
    for stream in ["single", "dual"]:
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


def eq3(dv, category, gain, gnorm, leakage, seq_length=10):
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
        The leakage parameter in the range [0, 1], where ``0`` means
        that each sample in the sequence of length `seq_length` receives
        a weight of ``1`` (no recency); and ``1`` means that all samples
        except the last in the sequence receive a weight of ``0``, and
        the last receives a weight of ``1`` (strongest possible recency).
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
    leakage_term = (1 - leakage) ** (seq_length - np.arange(1, seq_length + 1))
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
        rdm = rdm / rdm.max()
        assert np.isclose(rdm.min(), 0)
        assert np.isclose(rdm.max(), 1)
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
