"""Configure common variables and data locations."""

from pathlib import Path

server_dir = Path("/run/user/1000/gvfs/sftp:host=141.14.156.202,user=appelhoff/home/appelhoff/Projects/ARC-Studies/eeg_compression")
sourcedata = server_dir / "ecomp_data" / "sourcedata"
