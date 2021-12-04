"""Configure common variables and data locations."""

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
