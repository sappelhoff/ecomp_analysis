"""Configure common variables and data locations."""

import json
from pathlib import Path

DATA_DIR = Path("/run/user/1000/gvfs/sftp:host=141.14.156.202,user=appelhoff/home/appelhoff/Projects/ARC-Studies/eeg_compression/ecomp_data")


def get_sourcedata(sub, stream, data_dir=DATA_DIR):
    """Return path to source .vhdr and .tsv file for sub and stream."""
    sub_sourcedata = data_dir / "sourcedata" / f"sub-{sub:02}"
    substream = f"sub-{sub:02}_stream-{stream}"
    vhdr = sub_sourcedata / (substream + ".vhdr")
    tsv = sub_sourcedata / f"{stream}" / (substream + "_beh.tsv")
    return vhdr, tsv


def get_daysback(data_dir=DATA_DIR):
    """Get the 'daysback' for anonymizing this dataset."""
    with open(data_dir / "code" / "DAYSBACK.json", "r") as fin:
        data = json.load(fin)
    return data["daysback"]
