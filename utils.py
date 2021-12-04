"""Functions shared between scripts."""

import json
import warnings

import mne
import pandas as pd


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
        vhdr, tsv = get_sourcedata(sub, stream, data_dir)
        raw_stream = mne.io.read_raw_brainvision(vhdr, preload=True)
        raws.append(raw_stream)

    # put in order as recorded
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
    montage = mne.channels.make_standard_montage("standard_1020")
    raw = raw.set_montage(montage)

    # Add some recording info
    raw.info["line_freq"] = 50
    return raw
