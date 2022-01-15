"""Functions shared between scripts."""

import json
import pathlib
import warnings

import click
import mne
import pandas as pd

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
