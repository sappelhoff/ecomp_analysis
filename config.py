"""Configure common variables and data locations."""

from pathlib import Path

import numpy as np

# Path definitions
# -----------------------------------------------------------------------------

# The directory in which sourcedata, BIDS rawdata, and derivatives are nested
# - DATA_DIR_REMOTE_LOCAL is the directory on the server, accessed via ssh from local pc
# - DATA_DIR_LOCAL is the directory on the local pc
# - DATA_DIR_EXTERNAL is the directory on an external harddrive
# - DATA_DIR_REMOTE is the directory on the server
DATA_DIR_LOCAL = Path("/home/stefanappelhoff/Desktop/eComp")
DATA_DIR_EXTERNAL = Path(
    "/media/stefanappelhoff/LinuxDataAppelho/eeg_compression/ecomp_data/"
)
DATA_DIR_REMOTE = Path(
    "/home/appelhoff/Projects/ARC-Studies/eeg_compression/ecomp_data/"
)
start = "/run/user/1000/gvfs/sftp:host=141.14.156.202,user=appelhoff"
DATA_DIR_REMOTE_LOCAL = Path(
    start + "/home/appelhoff/Projects/ARC-Studies/eeg_compression/ecomp_data"
)

# The directory in which the analysis code is nested (this directory is tracked with git
# and is available on GitHub (private): https://github.com/sappelhoff/ecomp_analysis)
# same naming conventions as for DATA_DIR_* above
ANALYSIS_DIR_LOCAL = Path("/home/stefanappelhoff/Desktop/eComp/ecomp_analysis")
ANALYSIS_DIR_EXTERNAL = Path(
    "/media/stefanappelhoff/LinuxDataAppelho/eeg_compression/ecomp_data/"
)
ANALYSIS_DIR_REMOTE = Path(
    "/home/appelhoff/Projects/ARC-Studies/eeg_compression/ecomp_data/code/ecomp_analysis/"  # noqa: E501
)
ANALYSIS_DIR_REMOTE_LOCAL = None

# Constants
# -----------------------------------------------------------------------------
BAD_SUBJS = {
    15: "Consistently performed at chance level.",
    23: "Misunderstood response cues in one of the tasks.",
}

SUBJS = np.array(list(set(range(1, 33)) - set(BAD_SUBJS)))

DEFAULT_RNG_SEED = 42

OVERWRITE_MSG = "\nfile exists and overwrite is False:\n\n>>> {}\n"

# Threshold for epochs rejection in FASTER pipeline, step 2
# 3.29 corresponds to p < 0.001, two-tailed
FASTER_THRESH = 3.29
