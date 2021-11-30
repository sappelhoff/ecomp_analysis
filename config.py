"""Configure common variables and data locations."""

import json
import warnings
from pathlib import Path

# Path definitions
# -----------------------------------------------------------------------------

# Accessing data
start = "/run/user/1000/gvfs/sftp:host=141.14.156.202,user=appelhoff"
DATA_DIR_REMOTE = Path(
    start + "/home/appelhoff/Projects/ARC-Studies/eeg_compression/ecomp_data"
)

DATA_DIR_LOCAL = Path("/home/stefanappelhoff/Desktop/eComp")
DATA_DIR_EXTERNAL = Path(
    "/media/stefanappelhoff/LinuxDataAppelho/eeg_compression/ecomp_data/"
)

# Doing analysis
ANALYSIS_DIR = Path("/home/stefanappelhoff/Desktop/eComp/ecomp_analysis")

# Constants
# -----------------------------------------------------------------------------
BAD_SUBJS = {
    15: "Consistently performed at chance level.",
    23: "Misunderstood response cues in one of the tasks.",
}


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
